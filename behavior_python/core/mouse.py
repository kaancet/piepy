from .core import *
import matplotlib.pyplot as plt


class Mouse:
    def __init__(self, animalid: str, db_cols: dict = None, gsheet_init=True, **kwargs):
        # check if a Mouse object is already present
        self.id = animalid
        config = getConfig()
        self.db_path = config["databasePath"]
        self.db_interface = DataBaseInterface(self.db_path)

        if not os.path.exists(self.db_path):
            display(f"{self.db_path} does not exist, creating one now.")
        # this will either create a database or connect to an already existing one
        self.db_interface.connect()

        if self.db_interface.get_fields_info("animals"):
            self.db_cols = self.db_interface.get_fields_info("animals")
        else:
            # creates the animals table in the database
            # this should not happen
            self.db_cols = db_cols
            if self.db_cols is None:
                self.db_cols = {
                    "id": "VARCHAR(5)",
                    "age": "integer",
                    "framework": "text",
                    "currentStatus": "text",
                    "cage": "VARCHAR(5)",
                    "animalsInCage": "integer",
                    "sourceCage": "VARCHAR(5)",
                    "gender": "CHAR(1)",
                    "dob": "VARCHAR(6)",
                    "protocol": "VARCHAR(8)",
                    "strain": "text",
                    "color": "text",
                    "surgeryDate": "VARCHAR(6)",
                    "trainingStart": "VARCHAR(6)",
                    "deprivationType": "text",
                    "nSessions": "integer",
                }
                self.db_interface.set_table("animals", self.db_cols)

        if gsheet_init:
            self.init_from_gsheet()

    def __repr__(self) -> str:
        kws = [f"{key}={value!r}" for key, value in self.__dict__.items()]
        return "{}({})".format(type(self).__name__, ", ".join(kws))

    def save_animal(self) -> None:
        """Checks if an entry with  the mouse id exists, if not add it into the database"""
        self.db_interface.connect()

        mouse = self.db_interface.get_entries({"id": self.id}, "animals")
        if not mouse.empty:
            display("Entry for animal already exists")
            return 0
        else:
            mouse_entry = self.create_db_entry()
            mouse_entry["nSessions"] = 0
            self.db_interface.add_entry(mouse_entry, "animals")
            self.db_interface.commit()
            self.db_interface.close()

    def update_field(self, field: str, value) -> None:
        """Updates the given field with the given value for animal entry"""
        entry = self.db_interface.get_entries({"id": self.id}, "animals")
        if not entry.empty:
            if field not in self.db_interface.get_fields_info("animals").keys():
                raise ValueError(f"{field} field does not exist in the animals table")

            self.db_interface.update_entry({"id": self.id}, {field: value}, "animals")
        else:
            display(f"No entry in animals table can not update")

    def create_db_entry(self) -> dict:
        """Creates the database entry according to the database columns t"""
        return {k: v for k, v in self.__dict__.items() if k in self.db_cols.keys()}

    def init_from_gsheet(self) -> None:
        """Initializes a MOuse object from the entry in google sheets"""
        ignore = ["id"]  # ignore id because it's created in object intantiation
        # go to gsheet
        logsheet = GSheet("Mouse Log")
        # below, 0 is the log2021 sheet ID, 0 should always correspond to most recent sheet(year)
        sheet_df = logsheet.read_sheet(0)
        # find the row with animal id
        row = sheet_df[sheet_df["id"] == self.id].to_dict(orient="records")[0]
        # get the row values and set them as instance attributes
        for k, v in row.items():
            if k not in ignore:
                if v == "":
                    # if not filled prompt to fill
                    display(f" >>WARNING<< Property {k} is empty for {self.id}")
                    setattr(self, k, None)
                else:
                    if isinstance(v, str) and "-" in v:
                        # this is a date, turn it into lab date
                        try:
                            v = dt.strptime(v, "%d-%m-%Y").strftime("%y%m%d")
                        except:
                            pass
                    setattr(self, k, v)

    def read_gsheet(self) -> pd.DataFrame:
        """Reads the entries from the Mouse Database_new google sheet"""
        logsheet = GSheet("Mouse Database_new")
        sheet_df = logsheet.read_sheet(2)  # sheet id for log2021 is 2

        # convert decimal "," to "." and date string to datetime and drop na
        sheet_df["weight [g]"] = sheet_df["weight [g]"].apply(
            lambda x: str(x).replace(",", ".")
        )
        sheet_df["weight [g]"] = pd.to_numeric(sheet_df["weight [g]"], errors="coerce")
        sheet_df["supp water [µl]"] = pd.to_numeric(sheet_df["supp water [µl]"]).fillna(
            0
        )

        sheet_df["Date [YYMMDD]"] = sheet_df["Date [YYMMDD]"].apply(lambda x: str(x))
        sheet_df["Date [YYMMDD]"] = pd.to_datetime(
            sheet_df["Date [YYMMDD]"], format="%y%m%d"
        )

        rows = sheet_df[sheet_df["Mouse ID"] == self.id]
        if not len(rows):
            display(f"No entry in Mouse Database_new sheet for {self.id}")

        return rows

    def plot_life(self, **kwargs):
        """Plots the life of the animal with weight as line plot, consumed water as bars and stages of life as background vertical spans"""
        convert2date = ["dob", "lastSurgery", "retinotopyDate", "trainingStart"]
        fig = plt.figure(figsize=kwargs.get("figsize", (10, 5)))

        ax = fig.add_subplot(111)
        pass
