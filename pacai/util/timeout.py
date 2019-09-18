import multiprocessing
import sys
import traceback

REAP_TIME_SEC = 1

def invokeWithTimeout(timeout, function):
    """
    On timeout, success will be false and the value will be None.
    On error, success will be false and the value will be the string stacktrace.
    On successful completion,
    success will be true and the value wil be whatever the functions returns (or None).

    Returns: (success, function return value)
    """

    queue = multiprocessing.Queue(1)
    invokeHelper = lambda: _invokeHelper(queue, function)

    # Note that we use processes instead of threads so they can be more completely killed.
    process = multiprocessing.Process(target = invokeHelper)
    process.start()

    # Wait for at most the timeout.
    process.join(timeout)

    # Check to see if the thread is still running.
    if (process.is_alive()):
        # Kill the long-running thread.
        process.terminate()

        # Try to reap the thread once before just giving up on it.
        process.join(REAP_TIME_SEC)

        return (False, None)

    # Check to see if the process explicitly existed (like via sys.exit()).
    if (queue.empty()):
        return (False, 'Code explicitly exited (like via sys.exit()).')

    value, error = queue.get()

    if (error is not None):
        exception, stacktrace = error
        return (False, stacktrace)

    return (True, value)

def _invokeHelper(queue, function):
    value = None
    error = None

    try:
        value = function()
    except Exception as ex:
        error = (ex, traceback.format_exc())

    sys.stdout.flush()

    queue.put((value, error))
    queue.close()
