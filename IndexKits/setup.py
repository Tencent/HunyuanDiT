import re

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open("index_kits/__init__.py", "r") as file:
    regex_version = r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]'
    version = re.search(regex_version, file.read(), re.MULTILINE).group(1)


setup(  
    name='index_kits',  
    version=version,  
    author='jarvizhang',  
    author_email='jarvizhang@tencent.com',  
    description='An index kits for streaming reading arrow data.',  
    packages=['index_kits', 'index_kits/dataset'],
    scripts=['bin/idk'],
    install_requires=[
        "pillow>=9.3.0",
        "tqdm>=4.60.0",
        "pyarrow>=10.0.1",
        "torch>=1.9",
    ],  
    python_requires=">=3.8.12",
)
