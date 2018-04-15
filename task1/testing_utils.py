
def post_process_sentences(sentences, start_placeholder, end_placeholder):


    valid_submission_sentences=[]
    nb_valid_submissions=0

    for sentence in sentences:

        if sentence[-1] == end_placeholder:
            if (sentence[0] == start_placeholder):
                sentence.pop(0)

            valid_submission_sentences.append(' '.join(sentence))
            nb_valid_submissions=nb_valid_submissions+1

    print("Sentences predicted and found valid for submission are ", nb_valid_submissions)
    return valid_submission_sentences


def write_submission_predictions(sentences, start_placeholder, end_placeholder, group_number):

    valid_submission_sentences = post_process_sentences(sentences = sentences,start_placeholder=start_placeholder, end_placeholder=end_placeholder)
    with open("group"+group_number+".continuation.txt", 'a') as sub_file:
        for sentence in valid_submission_sentences:
            sub_file.write(sentence+"\n")
    print("Sentences of predictions written successfully to file")