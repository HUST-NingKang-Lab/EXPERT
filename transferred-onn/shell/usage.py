from time import ctime

# change package_name to your package name.
from package_name.version import VERSION


def run():
    cur_time = ctime()
    # change package_name to your package name.
    text = f"""
    # package_name
    
    Version {VERSION} ({cur_time} +0800)
    """
    print(text)
