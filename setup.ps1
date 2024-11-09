# Create project structure
$dirs = @(
    "notebooks",
    "src\models",
    "src\data",
    "src\utils",
    "src\training",
    "data\raw",
    "data\processed",
    "data\results\models",
    "data\results\plots",
    "data\results\logs",
    "tests",
    ".github\workflows",
    ".github\ISSUE_TEMPLATE"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path "D:\dlsacmp\$dir"
}

# Create __init__.py files
$initFiles = @(
    "src\__init__.py",
    "src\models\__init__.py",
    "src\data\__init__.py",
    "src\utils\__init__.py",
    "src\training\__init__.py",
    "tests\__init__.py"
)

foreach ($file in $initFiles) {
    New-Item -ItemType File -Force -Path "D:\dlsacmp\$file"
}

# Create requirements.txt
$requirements = @"
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
pandas>=1.3.0
scikit-learn>=0.24.0
wandb>=0.12.0
opencv-python>=4.5.0
einops>=0.3.0
tqdm>=4.62.0
Pillow>=8.0.0
jupyter>=1.0.0
pytest>=6.0.0
ipykernel>=6.0.0
"@

Set-Content -Path "D:\dlsacmp\requirements.txt" -Value $requirements

Write-Host "Project structure created successfully!"