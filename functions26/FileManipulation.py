# 2019-19-11 / updated on 2020-09-15
# This code was made for use in the Fu lab
# by Vasilis Niaouris

import matplotlib.pyplot as plt
import os
import re as string_match
import warnings


class FileManipulation:

    def __init__(self, file_no_list, file_type, file_no_format='02', folder_name='.', save_folder='.',
                 bool_save_figures=True):
        warnings.warn('The function FileManipulation is deprecated and will be deleted soon. Use FileNumberManager'
                      ' instead', DeprecationWarning)
        if isinstance(file_no_list, list):
            self.file_no_list = []
            self.file_name_list = []
            for file_no in file_no_list:
                if isinstance(file_no, int):
                    self.file_no_list.append(file_no)
                else:
                    print(str(file_no) + ' is not an integer')
        elif isinstance(file_no_list, int):
            self.file_name_list = []
            self.file_no_list = [file_no_list]
        else:
            print('This variable is neither a list or a single int.')
            self.file_name_list = None
            self.file_no_list = None

        self.file_type = file_type
        self.folder_name = folder_name
        self.file_no_format = file_no_format

        self.specific_type_file_name_list = []

        self.save_folder = save_folder
        self.bool_save_figures = bool_save_figures
        self.make_save_folder_if_needed()

    def get_specific_type_file_name_list(self):

        for file_name in os.listdir(self.folder_name):
            if file_name.endswith(self.file_type):
                self.specific_type_file_name_list.append(file_name)

    def get_file_list_names(self):

        if self.file_no_list is None or self.file_no_list == []:
            return None

        string_file_no_list = []
        for i in range(len(self.file_no_list)):
            string_file_no_list.append(format(self.file_no_list[i], self.file_no_format))
        string_file_no_list.sort()

        self.get_specific_type_file_name_list()
        for file in self.specific_type_file_name_list:
            for stfn in string_file_no_list:
                if len(string_match.findall(r"^" + stfn, file)):  # ^ means beginning of file. Look at regex for
                    # more info
                    self.file_name_list.append(file)
                    string_file_no_list.remove(stfn)

        return self.file_name_list

    def make_save_folder_if_needed(self):

        if not self.bool_save_figures:
            print('No request for saving data was made')
            return 0

        if self.save_folder == '.':
            print("No save folder specified. Will be saving at current directory: %s" % os.getcwd())
            return 1

        save_folder_exists = False
        for name in os.listdir():
            if name == self.save_folder:
                save_folder_exists = True
        if not save_folder_exists:
            try:
                os.mkdir(self.save_folder)
            except OSError:
                print("Creation of the directory %s in %s failed" % (self.save_folder, os.getcwd()))
            else:
                print("Successfully created the directory %s in %s" % (self.save_folder, os.getcwd()))
                return 2
        else:
            print("Folder %s already exists in %s" % (self.save_folder, os.getcwd()))

        return 1

    def save_many_figures(self, figure_no_list, save_figure_format):

        for figure_no in figure_no_list:
            plt.figure(figure_no)
            save_name = plt.gca().get_title()
            plt.savefig(self.save_folder + '/' + save_name + '.' + save_figure_format)
            print('Saved ' + self.save_folder + '/' + save_name + '.' + save_figure_format)

    def save_one_figure(self, figure_no, save_figure_format, custom_title=None):
            plt.figure(figure_no)
            save_name = custom_title
            if save_name is None:
                save_name = plt.gca().get_title()
            plt.savefig(self.save_folder + '/' + save_name + '.' + save_figure_format)
            print('Saved ' + self.save_folder + '/' + save_name + '.' + save_figure_format)

