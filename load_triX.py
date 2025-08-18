import numpy as np


# #------ Written by Tianxiong Han ------#
# #This file is for loading the data for triple-axis neutron spectroscopy
# #Function of normalize to counts per seconds is optional
# #May 3, 2024
# #--------------------------------------#

class import_triX_single:
    def __init__(self, instrument: str, exp: int, label_T: str):
        self.temp = None
        self.nor_to_cps = True
        # self.IPTS = IPTS
        self.exp = exp
        self.instrument = instrument
        self.temp_label = label_T
        self.mcu=0
        self.x = None
        self.y = None

    def load_data(self, path, run, nor_to_cps=True, name_x=None):
        list = []
        label = {'samplename': [], 'lattice constant': [], 'scan': [], 'x': [], 'y': [], 'mcu': [],
                 'temperature': [], 'tem_error': []}
        x = []
        y = []
        temperature = []
        yerr = []
        monitor = []
        # folder = path + 'IPTS-{}'.format(self.IPTS)
        folder = path
        str_exp = '{:04d}'.format(self.exp)
        str_run = '{:04d}'.format(run)
        file_name = (
                folder + '/exp{}/Datafiles/{}_exp{}_scan{}.dat'.format(self.exp, self.instrument, str_exp, str_run))
        with open(file_name, "r") as f:
            lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].split()
            if line[0] != '#':
                list.append(line)
            elif line[1] == 'latticeconstants':
                label['lattice constant'] = (line[3:])
            elif line[1] == 'def_x':
                if name_x is None:
                    label['x'] = line[-1]
                else:
                    label['x'] = name_x
            elif line[1] == 'def_y':
                label['y'] = line[-1]
            elif line[1] == 'samplename':
                label['samplename'] = line[-1]
            elif line[1] == 'scan':
                label['scan'] = line[-1]
            elif line[1] == 'Pt.':
                x_num = line.index(label['x']) - 1
                y_num = line.index(label['y']) - 1
                time_ind = line.index('time') - 1
                monitor_ind = line.index('monitor') - 1
                # mcu_ind = line.index('mcu') - 1
                t_sample_ind = line.index(self.temp_label) - 1
        for i in range(len(list)):
            if float(list[i][monitor_ind]) == 0:  ### remove the data point from abortion
                del list[i]
                print("Removed one point due to abortion")
                continue
            x.append(float(list[i][x_num]))
            monitor.append(float(list[i][monitor_ind]))
            temperature.append(float(list[i][t_sample_ind]))
        avg_monitor = np.average(monitor)
        # self.mcu = float(list[0][mcu_ind])
        if nor_to_cps == True:
            for i in range(len(list)):
                time = float(list[i][time_ind])
                y.append(float(list[i][y_num]) / monitor[i] * (avg_monitor / time))
                yerr.append(np.sqrt(float(list[i][y_num])) / monitor[i] * (avg_monitor / time))
            print("Data are normalized to counts per second using monitor!")
            print("Counted for {} seconds per point".format(time))
        elif nor_to_cps == False:
            for i in range(len(list)):
                time = float(list[i][time_ind])
                y.append(float(list[i][y_num]))
                yerr.append(np.sqrt(float(list[i][y_num])))
            self.nor_to_cps = False
            print("Data are NOT normalized to counts per second!")
            print("Counted for {} seconds per point".format(time))
            self.time = time
        avg_tem = round(np.average(temperature), 3)
        tem_error = round(np.std(temperature) / np.sqrt(np.size(temperature)), 3)
        label['temperature'] = avg_tem
        label['tem_error'] = tem_error
        # label['mcu'] = self.mcu
        print("read data successfully with {} lines, with sample temperature {}".format(len(list), avg_tem)
              + '\u00B1' + '{}K'.format(tem_error))
        self.temp = avg_tem
        self.x = np.array(x)
        self.y = np.array(y)
        self.yerr = np.array(yerr)
        return label, self.x, self.y, self.yerr

    def combine_runs(self,path,run_list, combine_variable, nor_to_cps=True, name_x=None):
        for idx, run_num in enumerate(run_list):
            run = import_triX_single()

        return self.x_list, self.y_list, self.yerr_list


# if __name__ == '__main__':
#     import_triX_single.combine_runs([],0)