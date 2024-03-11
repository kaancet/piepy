import datajoint as dj

dj.config['database.host'] = "localhost"
dj.config['database.user'] = "Dylan"
dj.config['database.password'] = "XiluDecy1/Pali"

dj.conn()
dj.config.save_global()
