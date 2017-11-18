import numpy as np
input_file = '../../data/traffic.label'
output_file = '../../data/traffic.pbtxt'

with open(output_file, 'w+') as output:
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.split():
                continue
            line = line.strip()
            number, name = line.split(' ', 1)
            output_str = "item { \n" + \
                        "  id: " + number + "\n" + \
                        "  name: \'" + name + '\'' + \
                        "\n}\n\n"
            # output_str = "item { \n \tid: " + number + "\n \t" + "name: " + name + "}\n"  
            # print(type(output_str))
            # exit(0)
            output.write(output_str)
        # if line.split():
        #     print(line, '\n')
        #     number, name = line.split()
            
        #     print(number, name)
print("Create success.")
