from setuptools import setup, find_packages
with open('requirements.txt') as f:
    required = f.read().splitlines()
setup(
   name='RHom',
   version='0.0.1',
   description='Package of functions to run and visualize PCA on ESQ data, and project between datasets.',
   author='Louis Chitiz',
   author_email='louischitiz@gmail.com',
   packages=find_packages(include=['RHom']),
   install_requires=required,
   package_data={'RHom': [
    'fonts/*.ttf'
    ]
    }
 
)
