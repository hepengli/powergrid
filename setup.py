from setuptools import setup, find_packages

setup(name='powergrid',
      version='0.0.1',
      description='Power Grid Environment',
      url='https://github.com/hepengli/powergrid',
      author='Hepeng Li',
      author_email='hepeng.li@maine.edu',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl', 'pandapower', 'pypower']
)
