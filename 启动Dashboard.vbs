Dim fso, sh, dir, q
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("WScript.Shell")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
q   = Chr(34)
sh.CurrentDirectory = dir
sh.Run "pythonw " & q & dir & "\dashboard\main.py" & q, 0, False
