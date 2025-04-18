import pandas as pd
import gspread
from googleapiclient import discovery
from oauth2client.service_account import ServiceAccountCredentials

from .config import config as cfg
from .io import display

# define the scope
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

cred_path = cfg.paths["gsheet"][0]


class GSheet:
    def __init__(self, spreadsheet_name, scope=scope, cred_path=cred_path):
        """
        Grants access to a google sheet to read/write
            spreadsheet_name: name of the spreadsheet to access
            scope:            scope of API's to be used
            cred_path:          path to the .json credential file
        """
        self.spreadsheet_name = spreadsheet_name
        self.scope = scope
        self.cred_path = cred_path

        self.creds = ServiceAccountCredentials.from_json_keyfile_name(
            self.cred_path, self.scope
        )
        self.client = gspread.authorize(self.creds)
        self.service = discovery.build("sheets", "v4", credentials=self.creds)

        self.sheet = self.access_sheet()
        display("Access granted to {0}".format(self.spreadsheet_name))

    def access_sheet(self):
        # add credentials to the account
        sheet = self.client.open(self.spreadsheet_name)
        return sheet

    def write_cell(self, cell, value):
        if type(value) is not list:
            value = [value]

        values = [value]
        body = {"values": values}
        try:
            result = (
                self.service.spreadsheets()
                .values()
                .update(
                    spreadsheetId=self.sheet.id,
                    range=cell,
                    valueInputOption="RAW",
                    body=body,
                )
                .execute()
            )
            # print('{0} cells updated'.format(result.get('updatedCells')))
            return 1
        except:
            return 0

    def read_cell(self, cell):
        # for reading a single cell
        result = (
            self.service.spreadsheets()
            .values()
            .get(spreadsheetId=self.sheet.id, range=cell)
            .execute()
        )
        return result["values"][0][0]

    def read_sheet(self, sheet_num):
        # sheet_num is 0 indexed sheet order
        si = self.sheet.get_worksheet(sheet_num)
        df = pd.DataFrame.from_dict(si.get_all_records())
        return df
