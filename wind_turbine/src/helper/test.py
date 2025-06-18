import zipfile
import os
from io import BytesIO

source_code_file = "src/agents/turbine_info.py"
_base_filename = source_code_file.split(".py")[0]

s = BytesIO()
z = zipfile.ZipFile(s, "w")
z.write(f"{source_code_file}")
z.close()