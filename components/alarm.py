

from playsound import playsound
from threading import Thread
from pathlib import Path
import time



#play sound in thread
global ROOT, alarm_name, sound_path
ROOT = str(Path(__file__).resolve().parents[1])
alarm_name = "/alert.mp3"

sound_path = ROOT + alarm_name

def play_sound():
    global sound_path
    print (sound_path)

    playsound(sound_path, False)

def run_alarm_in_thread():
    t = Thread(target=play_sound)
    
    t.start()
    time.sleep(1)
    t.join()
    return
    
    
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
