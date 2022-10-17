#coding=utf-8
#!/usr/bin/env python3
import xlrd
import numpy as np
import os

def get_data(filepath, watershed = None, dihindex = [], getname = False):
    path = os.getcwd() + "\\"  + filepath
    if not os.path.exists(path):
        print("file %s does not exist!" %(filepath))
        exit()
    try:
        workbook = xlrd.open_workbook(path)
        sheet0 = workbook.sheet_by_index(0)
        colenergy = sheet0.col_values(2)
        coldih = sheet0.col_values(1)
        colname = sheet0.col_values(0)
    except Exception:
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        colenergy = []
        coldih = []
        colname = []
        for i in range(10):
            line = lines[i].strip("\n").split(",")
        for line in lines:
            line = line.strip("\n").split(",")
            colenergy.append(line[2])
            coldih.append(line[1])
            colname.append(line[0])

    L = len(coldih)
    for i in range(L):
        tempdihs = coldih[i].strip(" \n").split(' ')
        for j in range(len(tempdihs)):
            tempdihs[j] = float(tempdihs[j])
        coldih[i] = tempdihs
        colenergy[i] = float(colenergy[i])
    if watershed:
        for i in reversed(range(0, L)):
            if colenergy[i] > watershed:
                colenergy.pop(i)
                coldih.pop(i)
                colname.pop(i)
    if dihindex != []:
        i = 0
        while i < len(coldih):
            coldih[i] = [coldih[i][index] for index in dihindex]
            i += 1
    if getname:
        return(np.array(coldih), np.array(colenergy), colname)
    else:
        return(np.array(coldih), np.array(colenergy))

def is_same(a, b, threshold, weight = None):
    if not weight:
        for i in range(len(a)):
            d = min(abs(a[i] - b[i]), (360 - abs(a[i] - b[i])))
            if d > threshold:
                return(False)
    else:
        for i in range(len(a)):
            d = min(abs(a[i] - b[i]), (360 - abs(a[i] - b[i]))) * weight[i]
            if d > threshold:
                return(False)
    return(True)

def rm_dplc_cmf(name, dihdata, energy, weight = [], threshold = 5):
    newcmfs = [dihdata[0]]
    newengs = [energy[0]]
    newname = [name[0]]
    weight = weight + [1 for i in range(len(weight), len(dihdata[0]))]
    for i, dihs in enumerate(dihdata):
        for j in range(1,len(newcmfs)+1):
            if newengs[-j] < energy[i] - 1:
                continue
            dihcet = newcmfs[-j]
            if is_same(dihs, dihcet, threshold, weight = weight):
                if energy[i] <= newengs[-j]:
                    newengs[-j] = energy[i]
                    newcmfs[-j] = dihs
                    newname[-j] = name[i]
                break
        else:
            newcmfs.append(dihs)
            newengs.append(energy[i])
            newname.append(name[i])
    print("find %d unique conformations from %d conformations!" %(len(newcmfs), len(dihdata)))
    return(newname, newcmfs, newengs)

def main(filepath):
    """
    二肽构象二面角去重程序。会将去重后的构象输出到*_unique.csv中
    """
    dihdata, energy, name = get_data(filepath, getname = True)
    name, dihdata, energy = rm_dplc_cmf(name, dihdata, energy, weight = [1,2,2,1], threshold = 5)
    f = open(filepath.replace(".", "_unique."), "w")
    for i in range(len(name)):
        f.write(name[i] + ",")
        f.write(" ".join(map(str, dihdata[i])))
        f.write("," + str(energy[i]) + "\n")
    f.close()

main("xls_databases\\VY.csv")