import sqlite3
import pandas as pd

def safe_str(var_string: str) -> str:
        if 'DROP' in var_string:
            raise ValueError('no.')
        return var_string

class DataBaseInterface:
    def __init__(self, db_path: str):
        self.db_path: str = db_path
        self.connection = None
        self.cursor = None

    def __repr__(self) -> None:
        kws = [f'{key}={value!r}' for key, value in self.__dict__.items()]
        return '{}({})'.format(type(self).__name__, ', '.join(kws))

    def connect(self,verbose:bool=True) -> None:
        """ Connects to a database file"""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
            if verbose:
                print(f'Connected to {self.db_path}')

    def get_fields_info(self, table_name: str) -> dict:
        """Returns the name and type of fields in a dictionary"""
        if self.cursor is None:
            self.connect()
            self.cursor = self.connection.cursor()

        c = self.cursor.execute(f'PRAGMA TABLE_INFO({safe_str(table_name)})')
        return {row[1]:row[2] for row in c}
        
    def set_table(self, table_name:str, table_cols:dict) -> None:
        """ Creates a table or if that table already exists, does nothing"""
        if self.cursor is None:
            self.connect()
            self.cursor = self.connection.cursor()
        try:    
            cols = """"""
            for k,v in table_cols.items():
                cols += f"""{safe_str(k)} {safe_str(v)},"""
            cols = cols[:-1]
            command = f"""CREATE TABLE {safe_str(table_name)} ({safe_str(cols)})"""
            self.cursor.execute(command)
            print(f'Created {table_name} table in {self.db_path}')
        except sqlite3.OperationalError:
            print(f'{table_name} table already exists, just access it')
        self.commit()
        self.close()

    def add_entry(self,entry:dict,table_name:str,verbose:bool=True) -> None:
        """ Adds an entry into the given table"""
        if self.cursor is None:
            self.connect(verbose=False)
            self.cursor = self.connection.cursor()

        if not self.exists(entry,table_name):
            command = f"INSERT INTO {safe_str(table_name)} VALUES"
            vals = "("
            for k,_ in entry.items():
                vals += f":{safe_str(k)}, "
            vals = vals[:-2]
            command += vals + ")"
            self.cursor.execute(command,entry)
            self.commit()
            self.close()
            if verbose:
                print(f'Added entry to {table_name}')
            
    def update_entry(self,entry:dict,update_dict:dict,table_name:str,verbose:bool=True) -> None:
        """ Updates an entry given as a dict with a single key:value pair. 
        The update_dict has the fields and values to be updated """
        if self.cursor is None:
            self.connect()
            self.cursor = self.connection.cursor()
        # table name
        command = f"""UPDATE {safe_str(table_name)} SET """
        
        # update key value pair
        for f in update_dict.keys():
            if f not in self.get_fields_info(table_name).keys():
                raise ValueError(f'{f} field not present in {table_name} table')
            command += f"""{safe_str(f)}=:{safe_str(f)},"""
        command = command[:-1]# get rid of final comma
        command += """ WHERE """ 
        
        # conditional key value pairs
        for k,v in entry.items():
            command += f"""{safe_str(str(k))}='{safe_str(str(v))}' AND """
        command = command[:-4] # remove the last AND
        self.cursor.execute(command,update_dict)
        if verbose:
            print(f'Sent command to update the entry in {table_name} table')
        self.commit()
        self.close()
        
    def remove_entry(self,entry:dict,table_name:str) -> None:
        """ Removes the entry that satisfies the key==value condition in the entry dict argument"""
        if isinstance(entry,dict):
            if len(entry)>1:
                raise ValueError(f'Entry search should be done with a single key:value pair dictionary')
            entry_field,entry_val = zip(*entry.items())
        elif isinstance(entry,str):
            if entry != 'all':
                raise ValueError(f'{entry} is not a valid value for entry use a dict or "all" to delete everyhting in the table')
        else:
            raise TypeError(f'{type(entry)} is not a valid type for entry argument')
        
        if self.cursor is None:
            self.connect()
            self.cursor = self.connection.cursor()
        
        if entry == 'all':
            command = f""" DELETE FROM {safe_str(table_name)} """
        else:
            command = f"""DELETE FROM {safe_str(table_name)} WHERE {safe_str(str(entry_field[0]))}='{safe_str(str(entry_val[0]))}'"""
        
        self.cursor.execute(command)
        print(f'Removed {entry} in {table_name} table')
        self.commit()
        self.close()
        
    def delete_table(self,table_name:str) -> None:
        """ Deletes the given table """
        if self.cursor is None:
            self.connect()
            self.cursor = self.connection.cursor()
            
        command = f""" DROP TABLE {safe_str(table_name)}"""
        self.cursor.execute(command)
        print(f'Deleted the {table_name} table!')
        self.commit()
        self.close()
    
    def exists(self,entry:dict,table_name:str) -> bool:
        """ Convenience wrapper fro checking if an entry exists """
        temp = self.get_entries(entry,table_name)
        if temp.empty:
            return 0
        else:
            return 1

    def get_entries(self, entry_dict:dict, table_name:str) -> pd.DataFrame:
        """ Returns a dataframe of entries that satisfy the field==value condition"""
    
        if self.cursor is None:
            self.connect(verbose=False)
            self.cursor = self.connection.cursor()
        
        table_fields = list(self.get_fields_info(table_name).keys())
        df = pd.DataFrame(columns=table_fields,dtype=object)

        command = f"""SELECT * FROM {safe_str(table_name)} WHERE """
        for k,v in entry_dict.items():
            if k not in table_fields:
                raise ValueError(f'{k} is not a field in {table_name} table')
            command +=  f""" {safe_str(k)} = :{safe_str(k)} AND"""
        command = command[:-3] # remove the last AND
        rows = self.cursor.execute(command,entry_dict)
        for r in rows:
            df = df.append(pd.Series(r,index=table_fields),ignore_index=True)
        
        return df
    
    def print_table(self, table_name: str) -> pd.DataFrame:
        """Returns the table as a dataframe """
        if self.cursor is None:
            self.connect()
            self.cursor = self.connection.cursor()
            
        cols_dict = self.get_fields_info(table_name)
        col_names = list(cols_dict.keys())
        table_df = pd.DataFrame(columns=col_names,dtype=object)
        for entry in self.cursor.execute(f'SELECT * FROM {safe_str(table_name)}'):
            table_df = table_df.append(pd.Series(entry,index=col_names),ignore_index=True)
        self.close()
        return table_df

    def commit(self) -> None:
        """ Commits the changes in the table/database"""
        if self.connection is not None:
            self.connection.commit()

    def close(self) -> None:
        """ Closes the connection to the database"""
        if self.connection is not None:
            self.connection.close()
            self.connection = None
            self.cursor = None
