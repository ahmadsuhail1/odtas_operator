

from playsound import playsound
from threading import Thread
import time
import asyncio


#play sound in thread


def play_sound():

    playsound('D:/Workspace/FYP/Development/campaign-manager/backend/alert.mp3', False)

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
