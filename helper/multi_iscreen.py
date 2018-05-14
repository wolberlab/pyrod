import os

last_file = 10
current_file = 0
actives = '/path/to/actives.ldb'
decoys = '/path/to/decoys.ldb'
directory = '/path/to/pharmacophore/library'

while current_file <= last_file:
    file_name = '/'.join([directory, str(current_file)])
    os.system('iscreen -q {0}.pml -d {1}:active,{2}:inactive -o {0}.sdf -l {0}.log -R {0}.png'.format(file_name,
                                                                                                      actives, decoys))
    try:
        with open('{}.log'.format(file_name, 'r')) as log_file:
            if 'iscreen finished successfully' in log_file.read():
                current_file += 1
    except FileNotFoundError:
        pass
