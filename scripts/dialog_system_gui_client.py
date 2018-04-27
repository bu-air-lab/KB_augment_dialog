#!/usr/bin/env python

from bwi_msgs.srv import QuestionDialog
import rospy


if __name__ == "__main__":
    rospy.wait_for_service('question_dialog')
    handle = rospy.ServiceProxy('question_dialog', QuestionDialog)

    res = handle(2, "How can I help you?", [], 2)

    print res.text

    rospy.sleep(2)
