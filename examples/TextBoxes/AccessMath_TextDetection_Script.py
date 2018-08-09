# ===============================================================================

#============================================================================
# Preprocessing Model for ST3D indexing - V 1.1
#
# Kenny Davila
# - Created:  March 10, 2017
#
#============================================================================

"""
make sure you are in ~/git/TextBoxes
export PYTHONPATH=/home/buralako/git/AccessMath_ICDAR_2017_code:./python
python examples/TextBoxes/AccessMath_TextDetection_Script.py /home/buralako/git/AccessMath_ICDAR_2017_code/test_data/db_AccessMath2015.xml -l lecture_02
"""
import sys
from AccessMath.preprocessing.config.parameters import Parameters
from AccessMath.preprocessing.user_interface.console_ui_process import ConsoleUIProcess
# from AccessMath.preprocessing.video_worker.text_detection import TextDetection
from AccessMath_TextDetection import TextDetection

def get_worker(process):
    worker = TextDetection(visualize=False,
                           detection_threshold=0.1)
    return worker

def get_results(worker):
    return (worker.text_bbox,)

def main():
    #usage check
    if not ConsoleUIProcess.usage_check(sys.argv):
        return

    process = ConsoleUIProcess(sys.argv[1], sys.argv[2:], None, Parameters.Output_TextDetection)
    if not process.initialize():
        return

    # fps = Parameters.Sampling_FPS
    fps = 15
    process.start_video_processing(fps, get_worker, get_results, 0, True)
    print("finished")

if __name__ == "__main__":
    main()
