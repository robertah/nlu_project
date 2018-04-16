from config import *



def task_params(task, experiment=None):

    training_with_w2v = False
    lstm_cell_state = lstm_cell_state_down
    down_project = False

    if task==1:
        if str(experiment) == 'A':
            None
        if str(experiment) == 'B':
            training_with_w2v = True
        if str(experiment) == 'C':
            training_with_w2v = False
            lstm_cell_state = 2 * lstm_cell_state_down
            down_project = True

    return(training_with_w2v, lstm_cell_state, down_project)

# def main():
#
#     training_with_w2v, lstm_cell_state, down_project = task_params(1)
#     print(training_with_w2v, lstm_cell_state, down_project)
#
# main()