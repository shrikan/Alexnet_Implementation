import pip

def install(package):
    pip.main(['install', package])

def install_all_packages(modules_to_try):
    for module in modules_to_try:
        try:
           __import__(module)
        except ImportError as e:
            install(e.name)

if __name__ == '__main__':
    modules_required = ["pandas", "sklearn", "xgboost", "matplotlib", "geopy"]