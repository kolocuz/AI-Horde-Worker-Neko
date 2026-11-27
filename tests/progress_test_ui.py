import curses
import time


STATE_IDLE = "Состояние: ожидание работы..."
STATE_WORKING = "Состояние: работаю над job d919feb1 (interrogation)"
STATE_DONE = "Состояние: работа завершена"
STATE_FAULT = "Состояние: ошибка / ожидание восстановления"


def clamp(value, low=0, high=100):
    return max(low, min(high, value))


def draw_bar(stdscr, y, x, width, percent, indeterminate=False, tick=0):
    width = max(10, width)
    inner = width - 2
    if indeterminate:
        block = 8
        pos = tick % max(1, inner - block + 1)
        chars = ["-"] * inner
        for i in range(pos, min(inner, pos + block)):
            chars[i] = "#"
        bar = "[" + "".join(chars) + "]"
    else:
        filled = round(inner * (clamp(percent) / 100))
        bar = "[" + ("#" * filled) + ("-" * (inner - filled)) + "]"
    stdscr.addnstr(y, x, bar, width)


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)

    state = STATE_IDLE
    detail = "Нет активной работы"
    progress = 0
    indeterminate = False
    tick = 0

    while True:
        tick += 1
        height, width = stdscr.getmaxyx()
        stdscr.erase()

        title = "Тест TUI прогресс-бара (q - выход)"
        controls = "1:нет работы  2:работаю  3:+10%  4:-10%  i:indeterminate  e:ошибка  r:сброс  q:выход"
        stdscr.addnstr(0, 2, title[: max(0, width - 4)], max(0, width - 4), curses.A_BOLD)
        stdscr.addnstr(2, 2, state[: max(0, width - 4)], max(0, width - 4))
        stdscr.addnstr(3, 2, detail[: max(0, width - 4)], max(0, width - 4))

        bar_y = 5
        bar_x = 2
        bar_w = max(20, min(width - 4, 70))
        draw_bar(stdscr, bar_y, bar_x, bar_w, progress, indeterminate=indeterminate, tick=tick)

        if indeterminate:
            pct = "в процессе..."
        else:
            pct = f"{int(progress)}%"
        stdscr.addnstr(bar_y, bar_x + bar_w + 2, pct[: max(0, width - (bar_x + bar_w + 4))], max(0, width - (bar_x + bar_w + 4)))
        stdscr.addnstr(7, 2, controls[: max(0, width - 4)], max(0, width - 4))
        stdscr.addnstr(9, 2, "Это тестовый TUI, не GUI. Кнопки = клавиши."[: max(0, width - 4)], max(0, width - 4))

        stdscr.refresh()

        try:
            key = stdscr.get_wch()
        except curses.error:
            key = None
        except Exception:
            key = None

        if key is None:
            continue
        if isinstance(key, str):
            k = key.lower()
        else:
            k = ""

        if k == "q":
            break
        elif k == "1":
            state = STATE_IDLE
            detail = "Нет активной работы"
            progress = 0
            indeterminate = False
        elif k == "2":
            state = STATE_WORKING
            detail = "Обработка формы: caption / nsfw / interrogation"
            indeterminate = False
            if progress >= 100:
                progress = 0
        elif k == "3":
            if indeterminate:
                indeterminate = False
            if progress == 0 and state == STATE_IDLE:
                state = STATE_WORKING
                detail = "Обработка формы: caption / nsfw / interrogation"
            progress = clamp(progress + 10)
            if progress >= 100:
                state = STATE_DONE
                detail = "Последний job завершён успешно"
            elif state not in (STATE_WORKING, STATE_DONE):
                state = STATE_WORKING
        elif k == "4":
            if indeterminate:
                indeterminate = False
            progress = clamp(progress - 10)
            if progress < 100 and state == STATE_DONE:
                state = STATE_WORKING
                detail = "Обработка формы: caption / nsfw / interrogation"
            if progress == 0 and state != STATE_FAULT:
                state = STATE_WORKING
                detail = "Обработка формы: caption / nsfw / interrogation"
        elif k == "i":
            indeterminate = not indeterminate
            if indeterminate:
                state = "Состояние: работаю..."
                detail = "Прогресс неопределённый (ожидание/обработка)"
            else:
                state = STATE_WORKING
                detail = "Возврат в режим определённого прогресса"
        elif k == "e":
            indeterminate = False
            progress = 0
            state = STATE_FAULT
            detail = "Последний job завершился с ошибкой, воркер ждёт следующую задачу"
        elif k == "r":
            indeterminate = False
            progress = 0
            state = STATE_IDLE
            detail = "Нет активной работы"


if __name__ == "__main__":
    curses.wrapper(main)
