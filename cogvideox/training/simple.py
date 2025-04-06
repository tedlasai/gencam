import signal
import time


def handle_signal(signum, frame):
    print(f"Signal {signum} received at {time.ctime()}")
    #create a file to indicate that the training was interrupted
    with open("/datasets/sai/gencam/cogvideox/interrupted.txt", "w") as f:
        f.write(f"Training was interrupted at {time.ctime()}")

if __name__ == "__main__":
    #args = get_args()

    print("Registering signal handler")
    #Register the signal handler (catch SIGUSR1)
    signal.signal(signal.SIGUSR1, handle_signal)
    #register kill signal

    x= 0
    while True:
        x+=1
        #print("X", x)
        time.sleep(1)
    #main(args)
