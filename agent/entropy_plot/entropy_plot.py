import matplotlib.pyplot as plt
import ast

def plot_entropy():

        f = open("added.txt", "r")
        s = f.read()
        added_point = ast.literal_eval(s)
        f.close()

        entropy_list = []
        f = open("entropy.txt", "r")
        for line in f:
            entropy_list.append(float(line))
        f.close()

        question_list = []
        f = open("question.txt", 'r')
        for line in f:
            question_list.append(line)
        f.close()

        answer_list = []
        f = open("answer.txt", 'r')
        for line in f:
            answer_list.append(line)
        f.close()        

        # plot the entropy
        w,h = plt.figaspect(0.3)
        fig=plt.figure(figsize=(w,h))
        ax = fig.add_subplot(111)
        plt.plot(entropy_list)
        plt.ylabel('Entropy of Dialog Belief', fontsize=16)
        plt.xlabel('Dialog Turn', fontsize=16)
        #plt.title('Entropy changes during a conversation')

        if added_point:
            x, y = added_point
            textpoint = (x+0.5, y-0.25)
            ax.annotate('Pop added', fontsize=14, xy=added_point, 
                arrowprops = dict(facecolor='black', shrink=0.05), xytext=textpoint)

        # labels (manual)
        
        x_list = [0, 1, 2, 3, 5, 7, 8, 9, 10]
        text_y_offset = [-0.13, 0.1, -0.3, 0.2, 0.0, 0.5, 0.5, 0.5, 0.5]
        text_x_offset = [0.75, 0.1, 0.0, 0.0, 0.4, 0, 0.0, 0.0, -0.3]
        n = 0
        for x in x_list:
            y = entropy_list[x]
            point = (x, y)
            textpoint = (x + text_x_offset[n], y + text_y_offset[n])
            if x < len(question_list) and x < len(answer_list):
                text = question_list[x] + answer_list[x].strip()
            else:
                break
            ax.annotate(
                text,
                fontsize=14, 
                xy=point, 
                arrowprops = dict(
                    facecolor='black', 
                    shrink=0.05, 
                    width=0.1, 
                    headwidth=0),
                xytext=textpoint,)
            n+=1

        #plt.grid()
        fig.savefig('entropy_plot.pdf')


if __name__ == '__main__':
    plot_entropy()