# cmd or powershell
python -m venv venv

# cmd 
venv\Scripts\activate.bat

# powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

# exit venv
deactivate

