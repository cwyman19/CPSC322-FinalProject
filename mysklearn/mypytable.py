from mysklearn import myutils

import copy
import csv
from tabulate import tabulate
'''Charlie Wyman and Jillian Berry
CPSC322 - Final Project
MyPyTable Class (from PA2)
'''

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """



    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """

        cols = len(self.column_names)
        rows = len(self.data) # note that we are not counting the header as a row in the table

        return rows, cols

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA") ****
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """

        if col_identifier not in self.column_names:
            raise ValueError(f"Invalid column name: '{col_identifier}'. Available columns: {self.column_names}")
        
        col_index = self.column_names.index(col_identifier)
        data_column = []
        x = 0
        for i in self.data:
            data_column.append(self.data[x][col_index])
            x += 1

        return data_column

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        
        for row in range(len(self.data)):
            for item in range(len(self.data[row])):
                try:
                    float(self.data[row][item])
                    self.data[row][item] = float(self.data[row][item])
                    if (self.data[row][item] % 1 == 0.0):
                        self.data[row][item] = int(self.data[row][item])
                except:
                    ValueError
                    
        pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        num_index = 0
        count = 0
        while (num_index < len(self.data)) and (count < len(row_indexes_to_drop)):
            if (num_index == row_indexes_to_drop[count] - count):
                self.data.pop(row_indexes_to_drop[count] - count)
                count += 1
                num_index -= 1
            
            num_index += 1
        pass

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load *****
        """

        with open(filename, mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            
            self.column_names = next(reader)
            for row in reader:
                self.data.append(row)
    
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

        pass

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """


        key_indexes = []

        # finding keys in the header line
        iterator = 0
        j = 0
        while iterator < len(self.column_names):
            if self.column_names[iterator] == key_column_names[j]:
                key_indexes.append(iterator)
                if (j < len(key_column_names) - 1):
                    j += 1
            iterator += 1


        #creating a dataset of only keys
        key_iterator = 0
        row_iterator = 0
        rows_iterator = 0
        key_row = []
        key_rows = []
        for row in self.data:
            for item in self.data[rows_iterator]:
                if (key_indexes[key_iterator] == row_iterator) and (key_iterator < len(key_indexes)):
                    key_row.append(self.data[rows_iterator][row_iterator])
                    if (key_iterator < len(key_indexes) - 1):
                        key_iterator += 1
                row_iterator += 1
            rows_iterator += 1
            row_iterator = 0
            key_iterator = 0
            key_rows.append(key_row)
            key_row = []

        #print(key_rows)
        is_duplicate = []
        i = 0
        while i < len(key_rows):
            is_duplicate.append(0)
            i += 1

        
        
        for row in range(len(key_rows) - 1):
            for item in range(row + 1, len(key_rows)):
                if (is_duplicate[item] == 0):
                    if (key_rows[row] == key_rows[item]):
                        is_duplicate[item] = 1
        
        #print("is_duplicate", is_duplicate)

        index_list = []
        for num in range(len(is_duplicate)):
            if is_duplicate[num] > 0:
                index_list.append(num)

        
                    

        return index_list

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """

        col_iterator = 0
        row_iterator = 0
        NA_flag = False
        while col_iterator < len(self.data):
            while (row_iterator < len(self.data[col_iterator])) and (NA_flag == False):
                if str(self.data[col_iterator][row_iterator]) == "NA":
                    self.data.pop(col_iterator)
                    NA_flag = True
                    col_iterator -= 1
                else:
                    row_iterator += 1
            row_iterator = 0
            NA_flag = False
            col_iterator += 1


        pass

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """

        column = self.get_column(col_name)
        total = 0
        count = 0
        NA_indexes = []
        column_index = self.column_names.index(col_name)
        for item in range(len(column)):
            if (column[item] == "NA"):
                NA_indexes.append(item)
            else:
                total += float(column[item])
                count += 1
        average = total / count

        for item in range(len(NA_indexes)):
            self.data[NA_indexes[item]][column_index] = round(average, 2) 


        pass 

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        stats_header = ["attribute", "min", "max", "mid", "avg", "median"]

        stats_data = []
        for item in range(len(col_names)):
            column = self.get_column(col_names[item])
            new_column = []
            for new_item in range(len(column)):
                if not(column[new_item] == "NA"):
                    new_column.append(float(column[new_item]))
            new_column = sorted(new_column)
            row = []
            attribute = col_names[item]
            total = 0
            count = 0
            for item in range(len(new_column)):
                total += int(new_column[item])
                count += 1
            minimum = min(new_column)
            maximum = max(new_column)
            mid = minimum + ((maximum - minimum) / 2)
            avg = total / count 
            if (len(new_column) % 2 == 0):
                median = (new_column[(len(new_column) // 2) - 1] + new_column[(len(new_column) // 2)]) / 2
            else:
                median = new_column[len(new_column) // 2]
            row.append(attribute)
            row.append(minimum)
            row.append(maximum)
            row.append(mid)
            row.append(avg)
            row.append(median)
            stats_data.append(row)
        




        return MyPyTable(stats_header, stats_data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """

        # finding keys in the primary header line
        key_indexes = []
        iterator = 0
        j = 0
        while iterator < len(self.column_names):
            if self.column_names[iterator] == key_column_names[j]:
                key_indexes.append(iterator)
                if (j < len(key_column_names) - 1):
                    j += 1
            iterator += 1
        
        #finding keys in the secondary header line
        other_key_indexes = []
        iterator = 0
        j = 0
        while iterator < len(other_table.column_names):
            if other_table.column_names[iterator] == key_column_names[j]:
                other_key_indexes.append(iterator)
                if (j < len(key_column_names) - 1):
                    j += 1
            iterator += 1
        
        # creating data set of just the keys in primary table
        key_iterator = 0
        row_iterator = 0
        rows_iterator = 0
        key_row = []
        table_1_keys = []
        for row in self.data:
            for item in self.data[rows_iterator]:
                if (key_indexes[key_iterator] == row_iterator) and (key_iterator < len(key_indexes)):
                    key_row.append(self.data[rows_iterator][row_iterator])
                    if (key_iterator < len(key_indexes) - 1):
                        key_iterator += 1
                row_iterator += 1
            rows_iterator += 1
            row_iterator = 0
            key_iterator = 0
            table_1_keys.append(key_row)
            key_row = []

        # creating data set of just keys in secondary table
        key_iterator = 0
        row_iterator = 0
        rows_iterator = 0
        key_row = []
        table_2_keys = []
        for row in other_table.data:
            for item in other_table.data[rows_iterator]:
                if (other_key_indexes[key_iterator] == row_iterator) and (key_iterator < len(other_key_indexes)):
                    key_row.append(other_table.data[rows_iterator][row_iterator])
                    if (key_iterator < len(other_key_indexes) - 1):
                        key_iterator += 1
                row_iterator += 1
            rows_iterator += 1
            row_iterator = 0
            key_iterator = 0
            table_2_keys.append(key_row)
            key_row = []
        
        other_header = other_table.column_names

        # creating a new header and table
        my_new_header = []
        for name in range(len(self.column_names)):
            my_new_header.append(self.column_names[name])
        other_data_indexes = []
        for name in range(len(other_header)):
            is_duplicate = False
            for col in range(len(self.column_names)):
                if other_header[name] == self.column_names[col]:
                    is_duplicate = True 
            if not(is_duplicate):
                my_new_header.append(other_header[name])
                other_data_indexes.append(other_header.index(other_header[name]))


        my_new_table = []
        for item in range(len(table_1_keys)):
            for row in range(len(table_2_keys)):
                is_missing = False
                if (table_1_keys[item] == table_2_keys[row]): #match
                    new_row = []
                    count = 0
                    for i in range(len(self.data[item])):
                        new_row.append(self.data[item][i])
                        if (self.data[item][i] == "NA"):
                            is_missing = True
                    for j in range(len(other_table.data[row])):
                        if (other_table.data[row].index(other_table.data[row][j]) == other_data_indexes[count]):
                            new_row.append(other_table.data[row][j])
                            if (count < len(other_data_indexes)):
                                count += 1
                            if ((other_table.data[row][j]) == "NA"):
                                is_missing = True
                    if not(is_missing):
                        my_new_table.append(new_row)

        return MyPyTable(my_new_header, my_new_table) 

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        key_indexes = []
        iterator = 0
        j = 0
        while iterator < len(self.column_names):
            if self.column_names[iterator] == key_column_names[j]:
                key_indexes.append(iterator)
                if (j < len(key_column_names) - 1):
                    j += 1
                    iterator = -1
            iterator += 1

        #print("table 1 keys indexes: ", key_indexes)
        
        
        #finding keys in the secondary header line
        other_key_indexes = []
        iterator = 0
        j = 0
        while iterator < len(other_table.column_names):
            if other_table.column_names[iterator] == key_column_names[j]:
                other_key_indexes.append(iterator)
                if (j < len(key_column_names) - 1):
                    j += 1
                    iterator = -1
            iterator += 1
        
        #print("table 2 keys indexes: ", other_key_indexes)
        
        # creating data set of just the keys in primary table
        key_iterator = 0
        row_iterator = 0
        rows_iterator = 0
        key_row = []
        table_1_keys = []
        for row in self.data:
            for item in self.data[rows_iterator]:
                if (key_indexes[key_iterator] == row_iterator) and (key_iterator < len(key_indexes)):
                    key_row.append(self.data[rows_iterator][row_iterator])
                    if (key_iterator < len(key_indexes) - 1):
                        key_iterator += 1
                row_iterator += 1
            rows_iterator += 1
            row_iterator = 0
            key_iterator = 0
            table_1_keys.append(key_row)
            key_row = []
        
        #print("table 1 keys: ", table_1_keys)

        # creating data set of just keys in secondary table
        key_iterator = 0
        row_iterator = 0
        rows_iterator = 0
        key_row = []
        table_2_keys = []
        for row in other_table.data:
            for item in other_table.data[rows_iterator]:
                if (other_key_indexes[key_iterator] == row_iterator) and (key_iterator < len(other_key_indexes)):
                    key_row.append(other_table.data[rows_iterator][row_iterator])
                    row_iterator = -1
                    if (key_iterator < len(other_key_indexes) - 1):
                        key_iterator += 1
                row_iterator += 1
            rows_iterator += 1
            row_iterator = 0
            key_iterator = 0
            table_2_keys.append(key_row)
            key_row = []

        #print("table 2 keys: ", table_2_keys)
        
        other_header = other_table.column_names

        # creating a new header and table
        my_new_header = []
        for name in range(len(self.column_names)):
            my_new_header.append(self.column_names[name])
        other_data_indexes = []
        for name in range(len(other_header)):
            is_duplicate = False
            for col in range(len(self.column_names)):
                if other_header[name] == self.column_names[col]:
                    is_duplicate = True 
            if not(is_duplicate):
                my_new_header.append(other_header[name])
                other_data_indexes.append(other_header.index(other_header[name]))
        
        #print("my new header: ", my_new_header)

        table1_used_rows_indexes = []
        table2_used_rows_indexes = []
        my_new_table = []
        for item in range(len(table_1_keys)):
            for row in range(len(table_2_keys)):
                if (table_1_keys[item] == table_2_keys[row]): #match
                    table1_used_rows_indexes.append(item)
                    table2_used_rows_indexes.append(row)
                    new_row = []
                    count = 0
                    for i in range(len(self.data[item])):
                        new_row.append(self.data[item][i])
                    for j in range(len(other_table.data[row])):
                        if (other_table.data[row].index(other_table.data[row][j]) == other_data_indexes[count]):
                            new_row.append(other_table.data[row][j])
                            if (count < len(other_data_indexes)):
                                count += 1
                    my_new_table.append(new_row)

    
        for rows in range(len(self.data)):
            if rows not in (table1_used_rows_indexes):
                new_row = []
                for column in range(len(my_new_header)):
                    if my_new_header[column] not in self.column_names:
                        new_row.append("NA")
                    else:
                        new_row.append(self.data[rows][self.column_names.index(my_new_header[column])])
                my_new_table.append(new_row)
                table1_used_rows_indexes.append(rows)

            
        for rows in range(len(other_table.data)):
            if rows not in table2_used_rows_indexes:
                new_row = []
                for column in range(len(my_new_header)):
                    if my_new_header[column] not in other_table.column_names:
                        new_row.append("NA")
                    else:
                        new_row.append(other_table.data[rows][other_table.column_names.index(my_new_header[column])])
                my_new_table.append(new_row)
                table2_used_rows_indexes.append(rows)
        

        return MyPyTable(my_new_header, my_new_table)
    
    def remove_column(self, column_index):
        '''Removes a column of data
        
        Args:
            column_index (int): index of column to be removed
        '''

        self.column_names.remove(self.column_names.index(column_index))

        for row in range(len(self.data)):
            self.data[row].remove(self.data[row][column_index])



