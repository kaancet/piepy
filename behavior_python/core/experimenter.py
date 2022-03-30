from sqlite3.dbapi2 import OperationalError
from ..utils import getConfig
from .dbinterface import DataBaseInterface

class Experimenter:
    __slots__ = ['name','first_name','last_name','position','nAnimals','db']
    def __init__(self,name:str,first_name:str,last_name:str,position:str,nAnimals:int=0) -> None:
        valid_positions = ['post-doc','intern','master student','technician','PhD student','PI','job student']
        self.name = name
        self.first_name = first_name
        self.last_name = last_name
        if position not in valid_positions:
            raise ValueError(f'{position} is not a valid position! Try one of:\n {valid_positions}')
        self.position = position
        self.nAnimals = nAnimals
        
        config = getConfig()
        self.db = DataBaseInterface(config['databasePath'])

    def __repr__(self) -> str:
        return f" {self.position} {self.first_name} {self.last_name} ({self.name}) with {self.nAnimals} animals"

    def add_experimenter(self) -> None:
        """ Adds the current Experimenter to the database"""
        db_dict = {k:getattr(self,k,None) for k in self.__slots__ if k != 'db'}
        try:
            self.db.add_entry(db_dict,'experimenters')
        except OperationalError:
            table_cols = {'name':'text',
                          'first_name':'text',
                          'last_name':'text',
                          'position':'text',
                          'nAnimals':'integer'}
            self.db.set_table('experimenters',table_cols=table_cols)
            self.db.add_entry(db_dict,'experimenters')
        
    def update_experimenter(self,attr:str,new_val) -> None:
        """ Updates the experimenter's single attribute
        (This will probably only be either the position or the nAnimals)"""
        if attr not in self.__slots__:
            raise ValueError(f'{attr} is not avalid attribute for Experimenters!')
        
        setattr(self,attr,new_val)
        db_dict = {k:getattr(self,k,None) for k in self.__slots__ if k != 'db'}
        self.db.update_entry({'name':self.name},db_dict,'experimenters')
    
    def add_nanimals(self,num_add:int) -> None:
        """ A convenience wrapper for update_experimenter to quickly add animals"""
        if not isinstance(num_add,int):
            raise ValueError("Can't add non-integer number of animals!")
        self.update_experimenter('nAnimals',self.nAnimals+num_add)