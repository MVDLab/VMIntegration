from setuptools import setup, find_packages


setup(
    name="hmpldat",
    version="0.0.1",
    author="Ian Zurutuza",
    author_email="ian.zurutuza@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1",
        "scipy",
        "numpy",
        "tensorflow==1.15",
        "opencv-python",
        "tqdm",
        "xlsxwriter",
        "matplotlib",
    ],
    url="https://github.com/ianzur/hmpldat",
    description="Data alignment tools for UNTHSC Human Movement Performance Lab",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
