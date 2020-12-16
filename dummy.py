import time


def add_stuff():
    return 2 + 2


def multiply_stuff():
    return 2 * 2


def divide_stuff():
    return 5 / 2


def subtract_stuff():
    return 5 - 2


def short_sleep():
    time.sleep(0.1)
    return


def medium_sleep():
    time.sleep(1)
    return


def long_sleep():
    time.sleep(5)
    return


def sleepy_day():
    short_sleep()
    medium_sleep()
    long_sleep()
    return


def somewhat_productive_day():
    short_sleep()
    add_stuff()
    subtract_stuff()
    long_sleep()
    return


def productive_day():
    multiply_stuff()
    divide_stuff()
    add_stuff()
    subtract_stuff()
    return


def my_week():

    # monday
    sleepy_day()
    # tuesday
    somewhat_productive_day()
    # wednesday
    somewhat_productive_day()
    # thursday
    productive_day()
    # friday
    sleepy_day()
    return


if __name__ == "__main__":
    my_week()
