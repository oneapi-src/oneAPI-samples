
BLUE = '34m'
message = 'hello world'


def display_colored_text(color, text):
    colored_text = f"\033[{color}{text}\033[00m"
    return colored_text

def main():
    print(display_colored_text(BLUE, message))
    
if __name__ == "__main__":
    main()
