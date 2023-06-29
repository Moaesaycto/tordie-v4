from utils import color
from settings import *

VERSION = "v4.0.0"

def init(short=False):
    print(f"\n{color.green}" + "{:=^40}".format(" TORDIE 4.0 ") + color.end)

    if not short:
        print(f"{color.yellow}Currently running: " + "{:.>22}".format(f" {VERSION} ") + color.end)
        print(f"{color.yellow}Developed by: " + "{:.>27}".format(" Moaesaycto (SL) ") + color.end)
        print(color.green + "{:=^40}".format("") + f"{color.end}\n")

    print(color.magenta + "Initializing program..." + color.end)
    if POINCARE_ERR < MAN_EPS or PARAM_REFLECT_M_DIFF < MAN_EPS or DIFF_STEP < MAN_EPS:
        print(f"{color.red}Custom machine number incompatible with settings.\nInitialization failed.{color.end}")
        exit()
    elif not short: print(f"{color.green}Custom machine number compatible.{color.end}")
    print(color.cyan + "Initialization complete." + color.end)