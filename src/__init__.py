from .module1 import function1, Class1
from .module2 import function2, Class2

# __init__.py

# Import specific modules or functions to make them accessible at the package level

# Define package-level variables
__version__ = "1.0.0"
__author__ = "Your Name"

# Optional: Perform package initialization
def initialize_package():
    print("Initializing the package...")