rom setuptools import setup, find_packages

with open('README.rst') as f:
    long_description = ''.join(f.readlines())


setup(
    name='speech-recognition',
    version='0.1',
    keywords='Speech Recogntion',
    description='End2End Speech Recognition',
    long_description=long_description,
    author='Adam Zvada',
    author_email='zvadaada@fit.cvut.cz',
    license='MIT',
    url='TODO',
    zip_safe=False,
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development',
        'Topic :: Utilities',
    ],
)