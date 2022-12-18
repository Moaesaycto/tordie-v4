class color:
    """
    COLOR CLASS

    Simple class for coloured messages

    \033[38;2;<r>;<g>;<b>m     #Select RGB foreground color
    \033[48;2;<r>;<g>;<b>m     #Select RGB background color
    """
    black = "\u001b[30m"
    red = "\u001b[31m"
    green = "\u001b[32m"
    yellow = "\u001b[33m"
    blue = "\u001b[34m"
    magenta = "\u001b[35m"
    cyan = "\u001b[36m"
    White = "\u001b[37m"
    end = "\033[0m"
