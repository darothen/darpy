from setuptools import setup

setup(
    name='marc-aie-utilities',
    version='0.1',
    py_modules=['marc_aie'],
    include_package_data=True,
    install_requires=[
        'click',
    ],
    entry_points='''
        [console_scripts]
        marc_aie=marc_aie:cli
    ''',
)
