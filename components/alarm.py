

from playsound import playsound
from threading import Thread
from pathlib import Path

import asyncio


#play sound in thread
ROOT = Path(__file__).resolve().parents[1]
alarm_name = "alert.mp3"

sound_path = Path(ROOT / "backend" / alarm_name)


def play_sound():

    playsound(sound_path, False)

async def run_alarm_in_thread():
    t = Thread(target=play_sound,daemon=True)
    t.start()
    await asyncio.sleep(1)
    t.join()
    print("Done")
    
    
# if "__main__" == __name__:    

#     t = Thread(target=play_sound)
#     t.start()
#     time.sleep(2)

#     t.join()

#     print("Done")

    

    

    # start_time = time.time()
    # while (time.time() - start_time) < 2:
    #     print("running")
# t = Thread(target=play_sound)
# t.daemon = True
# t.start()
