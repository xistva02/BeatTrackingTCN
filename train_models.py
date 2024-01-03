import subprocess

if __name__ == '__main__':

    versions_sr = ['5', '11', '22', '44']

    for version_sr in versions_sr:

        model_type = 'simple_tcn'
        version = '2020'
        cmd = f'python train.py' \
              f' --model_type {model_type} --version {version} --skip_output {0} --version_sr {version_sr}'
        subprocess.run(cmd)

        model_type = 'simple_tcn'
        version = '2020'
        cmd = f'python train.py' \
              f' --model_type {model_type} --version {version} --skip_output {1} --version_sr {version_sr}'
        subprocess.run(cmd)

        model_type = 'simple_tcn'
        version = 'dp'
        cmd = f'python train.py' \
              f' --model_type {model_type} --version {version} --skip_output {0} --version_sr {version_sr}'
        subprocess.run(cmd)

        model_type = 'simple_tcn'
        version = 'dp'
        cmd = f'python train.py' \
              f' --model_type {model_type} --version {version} --skip_output {1} --version_sr {version_sr}'
        subprocess.run(cmd)

        # bock 2020 model from scratch
        model_type = 'bock_2020'
        cmd = f'python train.py' \
              f' --model_type {model_type} --version_sr {version_sr}'
        subprocess.run(cmd)

        # bock 2019 model from scratch
        model_type = 'bock_2019'
        cmd = f'python train.py' \
              f' --model_type {model_type} --load_weights {0} --version_sr {version_sr}'
        subprocess.run(cmd)

        # bock 2019 model pretrained
        model_type = 'bock_2019'
        cmd = f'python train.py' \
              f' --model_type {model_type} --load_weights {1} --version_sr {version_sr}'
        subprocess.run(cmd)




