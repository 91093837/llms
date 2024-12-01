from nasdaq import get_data
from portfolio import build_portfolio


def main():
    data = get_data()
    build_portfolio(data)
    return None


if __name__ == "__main__":
    main()
