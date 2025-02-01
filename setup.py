from setuptools import setup, find_packages

setup(
    name="emonet",
    version="0.0.1",
    author="Param Uttarwar",
    author_email="param.uttarwar@casablanca.ai",
    description=" Emotion recognition using https://github.com/face-analysis/emonet",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/puttarwar/emonet",
    packages=find_packages(where="src"),
    include_package_data=True,
    package_dir={"": "src"},
    package_data={
        "emonet": ["*.pkl","*.pth","*.png",".gif" ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[ 'PyQt5', 'opencv-python-headless','torch',],

)
