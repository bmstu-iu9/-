import re
import argparse
import sys
from bitstring import *

def format_line(input_str):
    input_str = input_str.replace("\n", "")
    input_str = input_str.replace("\r\n", "")
    return input_str.replace(" ", "")

def close(f1, f2):
    f1.close()
    f2.close()
    return

genome_length = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assemble genome')
    parser.add_argument('-i','--input', help='Input file name',required=True)
    parser.add_argument('-o','--output',help='Output file name', required=True)
    args = parser.parse_args()

    f1 = open(args.input, 'r')
    f2 = open(args.output, 'wb')
    equal_pattern = re.compile('=')
    for line in f1:
        trimmed_line = format_line(line)
        if trimmed_line == "":
            continue
        splitted_string = re.split(equal_pattern, trimmed_line)
        if(len(splitted_string) != 2):
            print("Syntax error: Invalid syntax.")
            close(f1, f2)
            sys.exit(1)
        cond = splitted_string[0]
        operon = splitted_string[1]
        cond_values = re.split(",", cond)
        cur_cond = []
        cond_length = 0
        for v in cond_values:
            cur_cond_values = []
            substance = re.findall("\\d+", re.findall("\\[(\\d+)\\]", v)[0])
            sign = re.findall("[<>]",v)
            if(sign[0] == ">"):
                sign = 1
            elif(sign[0] == "<"):
                sign = 0
            else:
                print("Syntax error: Invalid Syntax.")
                close(f1,f2)
                sys.exit(1)
            threshold = re.findall("(\\d+)$", v)
            if((int)(substance[0]) > 127):
                print("Maximum substance value is 127.")
                close(f1, f2)
                sys.exit(1)
            cur_cond_values.append((int)(substance[0]))
            cur_cond_values.append(sign)
            if((int)(threshold[0]) > 127):
                print("Maximum threshold value is 127.")
                close(f1, f2)
                sys.exit(1)
            cur_cond_values.append((int)(threshold[0]))
            if(cond_length > 32):
                print("Maximum condition length is 32.")
                close(f1, f2)
                sys.exit(1)
            cur_cond.append(cur_cond_values)
            
        oper_values = re.split(",", operon)
        cur_oper = []
        oper_length = 0
        for v in oper_values:
            cur_oper_values = []
            substance = re.findall("\\d+", re.findall("\\[(\\d+)\\]", v)[0])
            sign = re.findall("[+-]",v)
            if(sign[0] == "-"):
                sign = 1
            elif(sign[0] == "+"):
                sign = 0
            else:
                print("Syntax error: Invalid Syntax.")
                close(f1, f2)
                sys.exit(1)
            rate = re.findall("(\\d+)$", v)
            if((int)(substance[0]) > 127):
                print("Maximum substance value is 127.")
                close(f1, f2)
                sys.exit(1)
            cur_oper_values.append((int)(substance[0]))
            cur_oper_values.append(sign)
            if((int)(rate[0]) > 127):
                print("Maximum rate value is 127.")
                close(f1, f2)
                sys.exit(1)
            cur_oper_values.append((int)(rate[0]))
            if(oper_length > 32):
                print("Maximum operon length is 32.")
                close(f1, f2)
                sys.exit(1)
            cur_oper.append(cur_oper_values)
        genome_length += 1
        if(genome_length > 32):
            print("Too big genome. MAX_GENOME_SIZE is 32.")
            close(f1, f2)
            sys.exit(1)
        print(cur_cond)
        print(cur_oper)
        for cond in cur_cond:
            substance = cond.pop(0)
            sign = cond.pop(0)
            threshold = cond.pop(0)
            f2.write(substance.to_bytes(1, byteorder = "big"))
            if(sign == 1):
                threshold *= 2
            f2.write(threshold.to_bytes(1, byteorder = "big"))
        for oper in cur_oper:
            substance = oper.pop(0)
            sign = oper.pop(0)
            rate = oper.pop(0)
            substance *= 2
            f2.write(substance.to_bytes(1, byteorder = "big"))
            if(sign == 1):
                rate *= 2
            f2.write(rate.to_bytes(1, byteorder = "big"))
    close(f1, f2)
    


    
