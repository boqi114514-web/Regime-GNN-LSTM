"""Daily stock opportunities: temporal encoder, dated multi-label graph and experts.

This is a small, actually trainable neural architecture, inspired by the explicit
concept aggregation of HIST and the learned routing of TRA, not a replication of
either paper. Stock sequences are encoded by a GRU; a bipartite stock/concept
message pass costs O(number of links * hidden), never O(number of stocks ** 2).
Four independently parameterised heads represent breakthrough, reversal,
continuation and range opportunities. Their soft router is learned jointly.

Return heads predict simple return in native units, risk heads predict forward
maximum adverse excursion, and rally heads predict fixed horizon events. No
stock identifiers, fixed concept embeddings or future memberships are learned.
All 20-session label ends are purged before fitting, including shorter targets.
The feature/daily execution builder remains responsible for adjustment factors,
next-open entry labels, missing/suspended sessions and corporate actions.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HORIZONS = (5, 10, 20)
RALLY_THRESHOLDS = (.10, .15, .30)
EXPERT_NAMES = ('breakthrough', 'reversal', 'continuation', 'range')
PROTOCOL = {
    'version': 'daily_graph_opportunity_v1',
    'frequency': 'Daily after-close predictions, earliest execution next open',
    'encoder': 'Shared stock GRU, stock -> dated multi-label concept -> stock mean aggregation',
    'experts': list(EXPERT_NAMES),
    'router': 'Learned softmax of own temporal, graph and market-mean embeddings; optional weak causal-stage supervision',
    'targets': {'horizons': list(HORIZONS), 'return': 'Native simple forward return',
                'downside': 'Nonnegative forward maximum adverse excursion',
                'rally_thresholds': list(RALLY_THRESHOLDS)},
    'label_purge': 'Every horizon label_end_date strictly before fit_cutoff; training labels additionally strictly before the first validation signal',
    'validation': 'Last 20 available pre-cutoff sessions by default; train-only scaler; early stopping and optional native-unit calibration',
    'score': 'Expected simple return minus explicit nonnegative downside risk penalty, NOT historical momentum or a ranking-score alias',
    'research_sources': ['https://arxiv.org/abs/2110.13716',
                         'https://arxiv.org/abs/2106.12950'],
}


def _day(value):
    text = str(value)[:10]
    if len(text) == 8 and text.isdigit():
        text = f'{text[:4]}-{text[4:6]}-{text[6:8]}'
    result = np.datetime64(text, 'D')
    if np.isnat(result):
        raise ValueError('A complete date is required')
    return result


def _tensor(value, *, device=None, dtype=torch.float32):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _matmul(matrix, dense):
    if matrix.layout == torch.strided:
        return matrix @ dense
    return torch.sparse.mm(matrix, dense)


def _sums(matrix, dimension):
    if matrix.layout == torch.strided:
        return matrix.sum(dim=dimension)
    # CSR supports mm but torch.sparse.sum does not; COO conversion is O(links).
    return torch.sparse.sum(matrix.to_sparse_coo(), dim=dimension).to_dense()


def concept_messages(encoded, membership):
    """Permutation-equivariant stock -> concept -> stock multi-label means.

    Membership weights must be finite and nonnegative. Empty concepts have no
    effect, and a stock with no known concepts receives a zero message. Concept
    identifiers are deliberately not parameters, so unseen dated concepts work.
    """
    membership = _tensor(membership, device=encoded.device)
    if membership.ndim != 2 or membership.shape[0] != encoded.shape[0]:
        raise ValueError('membership must have shape [stocks, concepts]')
    values = membership if membership.layout == torch.strided else membership.values() if membership.layout == torch.sparse_csr else membership.coalesce().values()
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError('membership weights must be finite and nonnegative')
    if membership.shape[1] == 0:
        return torch.zeros_like(encoded)
    if membership.layout == torch.sparse_coo:
        membership = membership.coalesce()
    transpose = membership.transpose(0, 1)
    if transpose.layout not in (torch.strided, torch.sparse_coo):
        transpose = transpose.to_sparse_coo().coalesce()
    theme = _matmul(transpose, encoded) / _sums(membership, 0).clamp_min(1e-12).unsqueeze(-1)
    return _matmul(membership, theme) / _sums(membership, 1).clamp_min(1e-12).unsqueeze(-1)


def build_causal_membership(stock_codes, links, signal_date):
    """Build a sparse graph only from links demonstrably available by signal.

    ``links`` is a DataFrame or iterable of dictionaries with ``ts_code``,
    ``theme_code`` and mandatory ``available_date``. Optional ``effective_from``
    and exclusive ``effective_to`` encode dated membership intervals. A current
    snapshot without historical availability dates is rejected, not backfilled.
    """
    day = _day(signal_date)
    records = links.to_dict('records') if hasattr(links, 'to_dict') else list(links)
    codes = [str(code) for code in stock_codes]
    if len(codes) != len(set(codes)):
        raise ValueError('stock_codes must be unique')
    stock_index = {code: number for number, code in enumerate(codes)}
    pairs = set()
    for row in records:
        if not all(key in row for key in ('ts_code', 'theme_code', 'available_date')):
            raise ValueError('Every membership link needs a historical available_date')
        available = _day(row['available_date'])
        if available > day:
            continue
        start = row.get('effective_from')
        end = row.get('effective_to')
        if start is not None and _day(start) > day:
            continue
        if end is not None and day >= _day(end):
            continue
        code, theme = str(row['ts_code']), str(row['theme_code'])
        if not theme or theme in ('None', 'nan'):
            raise ValueError('Theme identity must be present')
        if code in stock_index:
            pairs.add((stock_index[code], theme))
    themes = sorted({theme for _, theme in pairs})
    theme_index = {theme: number for number, theme in enumerate(themes)}
    coordinates = sorted((stock, theme_index[theme]) for stock, theme in pairs)
    indices = torch.tensor(coordinates, dtype=torch.long).T if coordinates else torch.empty((2, 0), dtype=torch.long)
    graph = torch.sparse_coo_tensor(indices, torch.ones(len(coordinates)),
                                    (len(codes), len(themes)), check_invariants=True).coalesce()
    return graph, themes


class DailyGraphOpportunityNet(nn.Module):
    """Learned temporal/graph mixture with 3 native-unit heads per expert."""

    def __init__(self, input_dim, hidden=32, sequence_length=20,
                 horizons=HORIZONS, experts=4):
        super().__init__()
        if input_dim < 1 or hidden < 2 or sequence_length < 2:
            raise ValueError('Positive feature, hidden and sequence dimensions are required')
        if tuple(horizons) != HORIZONS or experts != len(EXPERT_NAMES):
            raise ValueError('This protocol fixes horizons=(5,10,20) and four experts')
        self.input_dim, self.hidden = int(input_dim), int(hidden)
        self.sequence_length = int(sequence_length)
        self.horizons, self.experts = tuple(horizons), int(experts)
        self.temporal = nn.GRU(self.input_dim, self.hidden, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(self.hidden * 3, self.hidden),
                                    nn.LayerNorm(self.hidden), nn.SiLU())
        self.router = nn.Linear(self.hidden, self.experts)
        self.expert_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.SiLU(),
                          nn.Linear(self.hidden, len(self.horizons) * 3))
            for _ in range(self.experts)])
        for head in self.expert_heads:
            nn.init.normal_(head[-1].weight, std=.02)
            nn.init.zeros_(head[-1].bias)
            # Native risk units: initial softplus(-2.25)*.1 ~= one percent.
            with torch.no_grad():
                head[-1].bias[len(self.horizons):2 * len(self.horizons)].fill_(-2.25)

    def specification(self):
        return dict(input_dim=self.input_dim, hidden=self.hidden,
                    sequence_length=self.sequence_length,
                    horizons=self.horizons, experts=self.experts)

    def forward(self, sequences, membership):
        if sequences.ndim != 3 or sequences.shape[-1] != self.input_dim:
            raise ValueError('sequences must have shape [stocks, time, input_dim]')
        if sequences.shape[0] == 0 or sequences.shape[1] < self.sequence_length:
            raise ValueError('Each signal needs stocks and a complete trailing sequence')
        # Extra historical observations are ignored, never extra future ones.
        sequences = sequences[:, -self.sequence_length:]
        if not torch.isfinite(sequences).all():
            raise ValueError('All sequence features must be finite')
        _, final = self.temporal(sequences)
        own = final[-1]
        graph = concept_messages(own, membership)
        market = own.mean(dim=0, keepdim=True).expand_as(own)
        fused = self.fusion(torch.cat((own, graph, market), dim=-1))
        gate_logits = self.router(fused)
        gate = torch.softmax(gate_logits, dim=-1)
        raw = torch.stack([head(fused) for head in self.expert_heads], dim=1)
        h = len(self.horizons)
        expert_return = raw[:, :, :h]
        expert_downside = F.softplus(raw[:, :, h:2 * h]) * .1
        expert_rally = raw[:, :, 2 * h:]
        weights = gate.unsqueeze(-1)
        # Mixture probabilities (not mixture logits) are probabilistically valid.
        rally_probability = (weights * torch.sigmoid(expert_rally)).sum(dim=1)
        rally_logits = torch.logit(rally_probability.clamp(1e-6, 1 - 1e-6))
        return {'return_mu': (weights * expert_return).sum(dim=1),
                'downside': (weights * expert_downside).sum(dim=1),
                'rally_logits': rally_logits, 'gate': gate,
                'gate_logits': gate_logits, 'expert_return': expert_return,
                'expert_downside': expert_downside, 'expert_rally_logits': expert_rally}


@dataclass
class DailyOpportunityBatch:
    signal_date: object
    sequences: object
    membership: object
    returns: object = None
    downside: object = None
    label_end_dates: object = None
    stock_codes: object = None
    sequence_dates: object = None
    membership_asof: object = None
    stage_targets: object = None

    def validate(self, supervised=False):
        day = _day(self.signal_date)
        shape = tuple(self.sequences.shape)
        if len(shape) != 3 or shape[0] < 1 or shape[1] < 2 or shape[2] < 1:
            raise ValueError('Batch requires nonempty [stocks,time,features] sequences')
        graph_shape = tuple(self.membership.shape)
        if len(graph_shape) != 2 or graph_shape[0] != shape[0]:
            raise ValueError('Graph rows must match stock sequences')
        if self.stock_codes is not None and len(self.stock_codes) != shape[0]:
            raise ValueError('stock_codes must match stock rows')
        if self.sequence_dates is not None:
            dates = np.asarray(self.sequence_dates, dtype='datetime64[D]')
            if dates.shape not in ((shape[1],), (shape[0], shape[1])):
                raise ValueError('sequence_dates must match sequence time axis')
            if np.isnat(dates).any() or (dates > day).any() or (np.diff(dates, axis=-1) <= np.timedelta64(0, 'D')).any():
                raise ValueError('Sequences must contain strictly increasing past-only dates')
        if self.membership_asof is not None and _day(self.membership_asof) > day:
            raise ValueError('Future membership cannot be used at this signal')
        if supervised:
            expected = (shape[0], len(HORIZONS))
            for field in ('returns', 'downside', 'label_end_dates'):
                value = getattr(self, field)
                if value is None or tuple(np.shape(value)) != expected:
                    raise ValueError(f'{field} must have shape [stocks,3]')
            dates = np.asarray(self.label_end_dates, dtype='datetime64[D]')
            known = ~np.isnat(dates)
            if (dates[known] <= day).any():
                raise ValueError('Forward label dates must follow the signal')
        if self.stage_targets is not None:
            stages = np.asarray(self.stage_targets)
            if stages.shape != (shape[0],) or not np.isin(stages, (-1, 0, 1, 2, 3)).all():
                raise ValueError('stage_targets must be -1 (unknown) or 0..3 per stock')
        return self


def eligible_label_mask(batch, cutoff):
    """Only stocks with ALL finite, completed 5/10/20 targets are trainable."""
    batch.validate(supervised=True)
    dates = np.asarray(batch.label_end_dates, dtype='datetime64[D]')
    returns = np.asarray(batch.returns.detach().cpu() if isinstance(batch.returns, torch.Tensor) else batch.returns)
    downside = np.asarray(batch.downside.detach().cpu() if isinstance(batch.downside, torch.Tensor) else batch.downside)
    return ((~np.isnat(dates)).all(axis=1) & (dates < _day(cutoff)).all(axis=1)
            & np.isfinite(returns).all(axis=1) & np.isfinite(downside).all(axis=1)
            & (downside >= 0).all(axis=1))


def opportunity_loss(output, returns, downside, *, mask=None, stage_targets=None,
                     stage_weight=.05, balance_weight=.01):
    returns = _tensor(returns, device=output['return_mu'].device)
    downside = _tensor(downside, device=returns.device)
    if mask is None:
        mask = torch.isfinite(returns).all(-1) & torch.isfinite(downside).all(-1)
    else:
        mask = torch.as_tensor(mask, device=returns.device, dtype=torch.bool)
    if mask.shape != (returns.shape[0],) or not mask.any():
        raise ValueError('Loss needs at least one complete eligible stock target')
    r, d = returns[mask], downside[mask]
    if r.shape[-1] != 3 or not torch.isfinite(r).all() or not torch.isfinite(d).all() or (d < 0).any():
        raise ValueError('Loss targets must be finite native returns and nonnegative downside')
    # Normalize units for balanced learning, without clipping labels/returns.
    return_loss = F.smooth_l1_loss(output['return_mu'][mask] / .1, r / .1)
    risk_loss = F.smooth_l1_loss(output['downside'][mask] / .1, d / .1)
    thresholds = returns.new_tensor(RALLY_THRESHOLDS)
    rally_loss = F.binary_cross_entropy_with_logits(output['rally_logits'][mask], (r >= thresholds).float())
    gate = output['gate'][mask]
    balance_loss = ((gate.mean(dim=0) - 1 / len(EXPERT_NAMES)) ** 2).sum()
    stage_loss = returns.new_zeros(())
    if stage_targets is not None:
        targets = torch.as_tensor(stage_targets, device=returns.device, dtype=torch.long)
        known = mask & (targets >= 0) & (targets < len(EXPERT_NAMES))
        if known.any():
            stage_loss = F.cross_entropy(output['gate_logits'][known], targets[known])
    total = return_loss + .5 * risk_loss + .25 * rally_loss + stage_weight * stage_loss + balance_weight * balance_loss
    return {'total': total, 'return': return_loss, 'downside': risk_loss,
            'rally': rally_loss, 'stage': stage_loss, 'balance': balance_loss}


@dataclass
class FitConfig:
    hidden: int = 32
    sequence_length: int = 20
    epochs: int = 12
    learning_rate: float = .001
    weight_decay: float = .001
    validation_sessions: int = 20
    minimum_train_sessions: int = 20
    patience: int = 3
    seed: int = 42
    stage_weight: float = .05
    balance_weight: float = .01
    calibrate: bool = True
    gradient_clip: float = 2.


class _BatchSource:
    """Never retain stock sequences across dates: cache only label/date metadata."""

    def __init__(self, batches):
        self.batches = batches
        self.indexable = hasattr(batches, '__len__') and hasattr(batches, '__getitem__')
        if not self.indexable and not callable(batches):
            raise ValueError('Use a lazy indexable Sequence or a re-iterable batch_factory callable')

    def enumerate(self):
        if self.indexable:
            for index in range(len(self.batches)):
                yield index, self.batches[index]
        else:
            yield from enumerate(self.batches())


class BatchSelection:
    """Lazy iterable of (full cross-section batch, eligible-label mask)."""

    def __init__(self, source, records):
        self.source, self.records = source, records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if not self.source.indexable:
            raise TypeError('Factory selections support sequential iteration, not random access')
        record = self.records[index]
        return self.source.batches[record['index']], record['mask']

    def __iter__(self):
        if self.source.indexable:
            for record in self.records:
                yield self.source.batches[record['index']], record['mask']
        else:
            selected = {record['index']: record for record in self.records}
            for index, batch in self.source.enumerate():
                if index in selected:
                    record = selected[index]
                    if _day(batch.signal_date) != record['day']:
                        raise ValueError('batch_factory changed order/dates between passes')
                    yield batch, record['mask']


def purged_split(batches, fit_cutoff, config):
    """Chronological holdout; inputs may be a lazy Sequence or batch_factory."""
    cutoff = _day(fit_cutoff)
    source = _BatchSource(batches)
    records = []
    for index, batch in source.enumerate():
        day = _day(batch.signal_date)
        if day >= cutoff:
            continue
        mask = eligible_label_mask(batch, cutoff)
        if mask.any():
            dates = np.asarray(batch.label_end_dates, dtype='datetime64[D]')
            records.append(dict(index=index, day=day, mask=mask,
                                label_max=dates.max(axis=1)))
    records.sort(key=lambda record: record['day'])
    if len({record['day'] for record in records}) != len(records):
        raise ValueError('One complete cross-sectional batch per signal date is required')
    holdout = max(0, int(config.validation_sessions))
    if holdout and len(records) <= holdout:
        raise ValueError('Not enough past sessions for the requested validation holdout')
    validation_records = records[-holdout:] if holdout else []
    train_cutoff = validation_records[0]['day'] if validation_records else cutoff
    train_records = []
    for record in records:
        if record['day'] < train_cutoff:
            mask = record['mask'] & (record['label_max'] < train_cutoff)
            if mask.any():
                train_records.append({**record, 'mask': mask})
    if len(train_records) < config.minimum_train_sessions:
        raise ValueError('Not enough fully purged training sessions')
    return BatchSelection(source, train_records), BatchSelection(source, validation_records), train_cutoff


def _fit_scaler(training):
    count, total, squared = 0, None, None
    for batch, _ in training:
        values = _tensor(batch.sequences).reshape(-1, batch.sequences.shape[-1]).double().cpu()
        if not torch.isfinite(values).all():
            raise ValueError('Training feature sequences must be finite')
        count += values.shape[0]
        sums, squares = values.sum(0), (values * values).sum(0)
        total = sums if total is None else total + sums
        squared = squares if squared is None else squared + squares
    mean = total / count
    std = (squared / count - mean.square()).clamp_min(1e-8).sqrt()
    return mean.float(), std.float()


def _batch_forward(model, batch, mean, std, device):
    batch.validate()
    values = _tensor(batch.sequences, device=device)
    values = (values - mean.to(device)) / std.to(device)
    return model(values, _tensor(batch.membership, device=device))


def _calibration(model, validation, mean, std, device):
    calibration = dict(return_slope=np.ones(3), return_intercept=np.zeros(3),
                       downside_slope=np.ones(3), downside_intercept=np.zeros(3),
                       rally_temperature=1.)
    if not validation:
        return calibration
    predictions, observed, risks, observed_risks, rally_logits = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch, mask in validation:
            output = _batch_forward(model, batch, mean, std, device)
            predictions.append(output['return_mu'].cpu().numpy()[mask])
            risks.append(output['downside'].cpu().numpy()[mask])
            observed.append(np.asarray(batch.returns)[mask])
            observed_risks.append(np.asarray(batch.downside)[mask])
            rally_logits.append(output['rally_logits'].cpu().numpy()[mask])
    for prefix, xs, ys in [('return', predictions, observed), ('downside', risks, observed_risks)]:
        x, y = np.concatenate(xs), np.concatenate(ys)
        xc, yc = x - x.mean(0), y - y.mean(0)
        # Shrink towards native-unit identity and avoid unstable near-flat slopes.
        slope = ((xc * yc).sum(0) + .25) / ((xc * xc).sum(0) + .25)
        slope = np.clip(slope, 0., 2.)
        calibration[prefix + '_slope'] = slope
        calibration[prefix + '_intercept'] = np.clip(y.mean(0) - slope * x.mean(0), -.25, .25)
    logits = torch.from_numpy(np.concatenate(rally_logits))
    labels = torch.from_numpy((np.concatenate(observed) >= np.array(RALLY_THRESHOLDS)).astype(np.float32))
    temperatures = (.5, 1., 2.)
    losses = [float(F.binary_cross_entropy_with_logits(logits / temperature, labels)) for temperature in temperatures]
    calibration['rally_temperature'] = temperatures[int(np.argmin(losses))]
    return calibration


class FittedOpportunityModel:
    def __init__(self, network, mean, std, calibration=None, audit=None):
        self.network = network
        self.mean, self.std = mean.detach().cpu(), std.detach().cpu()
        self.calibration = calibration or dict(return_slope=np.ones(3), return_intercept=np.zeros(3),
            downside_slope=np.ones(3), downside_intercept=np.zeros(3), rally_temperature=1.)
        self.audit = audit or {}

    def predict(self, batch, device=None):
        if self.audit.get('fit_cutoff') and _day(batch.signal_date) < _day(self.audit['fit_cutoff']):
            raise ValueError('Cannot predict a signal before this checkpoint fit cutoff')
        device = device or next(self.network.parameters()).device
        self.network.to(device).eval()
        with torch.no_grad():
            output = _batch_forward(self.network, batch, self.mean, self.std, device)
        result = {name: value.detach().cpu().numpy() for name, value in output.items()}
        result['return_mu'] = result['return_mu'] * self.calibration['return_slope'] + self.calibration['return_intercept']
        result['downside'] = np.maximum(0., result['downside'] * self.calibration['downside_slope'] + self.calibration['downside_intercept'])
        result['rally_logits'] = result['rally_logits'] / self.calibration['rally_temperature']
        result['rally_probability'] = torch.sigmoid(torch.from_numpy(result['rally_logits'])).numpy()
        return result

    def save(self, path):
        payload = {'version': PROTOCOL['version'], 'specification': self.network.specification(),
                   'state_dict': self.network.state_dict(), 'mean': self.mean, 'std': self.std,
                   'calibration': {key: torch.as_tensor(value) for key, value in self.calibration.items()},
                   'audit': self.audit}
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path, device='cpu'):
        payload = torch.load(Path(path), map_location=device, weights_only=True)
        if payload.get('version') != PROTOCOL['version']:
            raise ValueError('Checkpoint protocol mismatch')
        network = DailyGraphOpportunityNet(**payload['specification']).to(device)
        network.load_state_dict(payload['state_dict'])
        calibration = {key: value.cpu().numpy() for key, value in payload['calibration'].items()}
        return cls(network, payload['mean'], payload['std'], calibration, payload['audit'])


def fit_walk_forward(batches, fit_cutoff, config=None, device='cpu', progress=None):
    """Fit one genuinely learned, strictly past-only checkpoint for daily use.

    A batch always contains its full known-at-signal cross-section for graph
    messages; only masked stocks contribute completed training targets. No
    future feature sequences or label values influence training/normalization.
    """
    config = config or FitConfig()
    if config.epochs < 1 or config.patience < 1 or config.learning_rate <= 0:
        raise ValueError('Fit epochs, patience and learning rate must be positive')
    training, validation, train_cutoff = purged_split(batches, fit_cutoff, config)
    if progress is not None:
        progress(dict(event='fit_start', fit_cutoff=str(_day(fit_cutoff)),
            train_sessions=len(training), validation_sessions=len(validation),
            train_rows=int(sum(record['mask'].sum() for record in training.records)),
            device=str(device)))
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValueError('Requested CUDA device is unavailable')
        torch.cuda.manual_seed_all(config.seed)
    mean, std = _fit_scaler(training)
    first_batch = next(iter(training))[0]
    model = DailyGraphOpportunityNet(first_batch.sequences.shape[-1], config.hidden,
                                    config.sequence_length).to(device)
    del first_batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    history, best, best_state, bad_epochs = [], float('inf'), None, 0
    for epoch in range(config.epochs):
        model.train()
        train_total, train_rows = 0., 0
        if training.source.indexable:
            order = list(range(len(training)))
            random.Random(config.seed + epoch).shuffle(order)
            epoch_batches = (training[index] for index in order)
        else:
            epoch_batches = iter(training)
        for batch, mask in epoch_batches:
            optimizer.zero_grad(set_to_none=True)
            output = _batch_forward(model, batch, mean, std, device)
            losses = opportunity_loss(output, batch.returns, batch.downside, mask=mask,
                stage_targets=batch.stage_targets, stage_weight=config.stage_weight,
                balance_weight=config.balance_weight)
            if not torch.isfinite(losses['total']):
                raise ValueError('Nonfinite training loss')
            losses['total'].backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite training gradient')
            optimizer.step()
            rows = int(mask.sum())
            train_total += float(losses['total'].detach()) * rows
            train_rows += rows
        model.eval()
        validation_total, validation_rows = 0., 0
        with torch.no_grad():
            for batch, mask in validation:
                output = _batch_forward(model, batch, mean, std, device)
                loss = opportunity_loss(output, batch.returns, batch.downside, mask=mask,
                    stage_targets=batch.stage_targets, stage_weight=config.stage_weight,
                    balance_weight=config.balance_weight)['total']
                rows = int(mask.sum())
                validation_total += float(loss) * rows
                validation_rows += rows
        train_loss = train_total / train_rows
        val_loss = validation_total / validation_rows if validation_rows else None
        history.append(dict(epoch=epoch + 1, train_loss=train_loss, validation_loss=val_loss))
        if progress is not None:
            progress(dict(event='epoch', fit_cutoff=str(_day(fit_cutoff)),
                          **history[-1]))
        objective = val_loss if val_loss is not None else train_loss
        if objective < best - 1e-8:
            best, best_state, bad_epochs = objective, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if validation and bad_epochs >= config.patience:
                break
    model.load_state_dict(best_state)
    calibration = _calibration(model, validation, mean, std, device) if config.calibrate else None
    audit = dict(protocol=PROTOCOL['version'], fit_cutoff=str(_day(fit_cutoff)),
                 train_cutoff=str(train_cutoff), configuration=asdict(config),
                 train_signal_dates=[str(record['day']) for record in training.records],
                 validation_signal_dates=[str(record['day']) for record in validation.records],
                 train_rows=int(sum(record['mask'].sum() for record in training.records)),
                 validation_rows=int(sum(record['mask'].sum() for record in validation.records)),
                 maximum_train_label_end=str(max(record['label_max'][record['mask']].max() for record in training.records)),
                 batch_access='lazy random-access Sequence' if training.source.indexable else 're-iterable factory',
                 history=history, device=str(device), calibration_from='purged historical validation only' if validation and config.calibrate else 'none')
    if progress is not None:
        progress(dict(event='fit_complete', fit_cutoff=audit['fit_cutoff'],
            epochs_completed=len(history), best_objective=best,
            train_rows=audit['train_rows'], validation_rows=audit['validation_rows']))
    return FittedOpportunityModel(model, mean, std, calibration, audit)


def expected_risk_score(prediction, horizon=20, risk_weight=.75):
    """Native expected return/risk tradeoff for the downstream lot optimizer."""
    if horizon not in HORIZONS or not np.isfinite(risk_weight) or risk_weight < 0:
        raise ValueError('Known horizon and nonnegative risk_weight required')
    index = HORIZONS.index(horizon)
    mu = np.asarray(prediction['return_mu'])[:, index]
    risk = np.asarray(prediction['downside'])[:, index]
    if not np.isfinite(mu).all() or not np.isfinite(risk).all() or (risk < 0).any():
        raise ValueError('Finite native predictions and nonnegative risks required')
    return mu - risk_weight * risk
