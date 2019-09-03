# -*- coding: utf-8 -*-

class sentence:
    def __init__(self):
        self.word = []
        self.tag = []
        self.wordchars = []

class dataset:
    def __init__(self):
        self.sentences = []
        self.name = ""
        self.total_word_count = 0
    
    def open_file(self, inputfile):
        self.inputfile = open(inputfile, mode = 'r', encoding='utf-8')
        self.name = inputfile.split('.')[0]

    def close_file(self):
        self.inputfile.close()

    def read_data(self, sentenceLen):
        wordCount = 0
        sentenceCount = 0
        sen = sentence()
        for s in self.inputfile:
            if(s == '\r\n' or s == '\n'):
                sentenceCount += 1
                self.sentences.append(sen)
                sen = sentence()
                if(sentenceLen !=-1 and sentenceCount >= sentenceLen):
                    break
                continue
            list_s = s.split('\t')
            str_word = list_s[1]#.decode('utf-8')
            str_tag = list_s[3]
            list_wordchars = list(str_word)
            sen.word.append(str_word)
            sen.tag.append(str_tag)
            sen.wordchars.append(list_wordchars)
            wordCount += 1
        self.total_word_count = wordCount
        print(self.name + ".conll contains " + str(len(self.sentences)) + " sentences")
        print(self.name + ".conll contains " + str(self.total_word_count) + " words")
        
class global_linear_model:
    def __init__(self):
        self.feature_dict = {}
        self.feature_keys = []
        self.feature_values = []
        self.feature_length = 0
        self.tag_dict = {}
        self.tag_length = 0
        self.update_times =[] 
        self.v = []
        self.w = []
        self.train = dataset()
        self.dev = dataset()

        self.train.open_file("train.conll")
        self.train.read_data(-1)
        self.train.close_file()

        self.dev.open_file("dev.conll")
        self.dev.read_data(-1)
        self.dev.close_file()
        
    def create_feature(self, sen, pos, left_tag):
        f = []
        
        tim1 = left_tag
        f.append("01:" + tim1)
            
        wi = sen.word[pos]
        f.append("02:" + wi)
        
        if(pos == 0):
            wim1 = "$$"
        else:
            wim1 = sen.word[pos - 1]
        f.append("03:" + wim1)
        
        len_sen = len(sen.word)
        if(pos == len_sen - 1):
            wip1 = "##"
        else:
            wip1 = sen.word[pos + 1]
        f.append("04:" + wip1)
        
        cim1m1 = wim1[-1]
        f.append("05:" + wi + cim1m1)
        
        cip10 = wip1[0]
        f.append("06:" + wi + cip10)
        
        ci0 = wi[0]
        f.append("07:" + ci0)
        
        cim1 = wi[-1]
        f.append("08:" + cim1)
        
        len_str = len(wi)
        for k in range(1, len_str - 1):
            cik = wi[k]
            f.append("09:" + cik)
            f.append("10:" + ci0 + cik)
            f.append("11:" + cim1 + cik)
            
        if(len_str == 1):
            f.append("12:" + wi + cim1m1 + cip10)
            
        for k in range(len_str - 1):
            cik = wi[k]
            cikp1 = wi[k + 1]
            if(cik == cikp1):
                f.append("13:" + cik + "consecutive")
        
        for k in range(1, len_str):
            if k > 4:
                break
            f.append("14:" + wi[:k])
            f.append("15:" + wi[-k:])
        #print(pos, f)
        return f
            
    def create_feature_space(self):
        for sen in self.train.sentences:
            for pos in range(len(sen.word)):
                if(pos == 0):
                    left_tag = "START"
                else:
                    left_tag = sen.tag[pos]
                f = self.create_feature(sen, pos, left_tag)
                for feature in f:
                    if feature not in self.feature_dict:
                        self.feature_dict[feature] = len(self.feature_dict)
                
                tag = sen.tag[pos]
                if tag not in self.tag_dict:
                    self.tag_dict[tag] = len(self.tag_dict)
                    
        self.feature_length = len(self.feature_dict)
        self.tag_length = len(self.tag_dict)
        self.feature_keys = list(self.feature_dict.keys())
        self.feature_values = list(self.feature_dict.values())
        
        self.w = [0]*(self.feature_length * self.tag_length)
        self.v = [0]*(self.feature_length * self.tag_length)
        self.update_times =[0]*(self.feature_length * self.tag_length)
        
        print("the total number of features is " + str(self.feature_length))
        print("the total number of tags is " + str(self.tag_length))
        
        #for i in range(100):
        #    print(self.feature_keys[i], self.feature_values[i])
        
    def get_feature_id(self, fv):
        fv_id = []
        for f in fv:
            if f in self.feature_dict:
                fv_id.append(self.feature_dict[f])
            #else:
                #print("not in")
        return fv_id
        
        
    def dot(self, fv_id, offset, flag):
        score = 0
        if(flag == "w"):
            for f_id in fv_id:
                score += self.w[f_id + offset]
        else:
            for f_id in fv_id:
                score += self.v[f_id + offset]
        return score
                
    def max_tag(self, sen, pos, flag):
        max_score = float("-Inf")
        max_tag = ""
        if pos == 0:
            left_tag = "START"
        else:
            left_tag = sen.tag[pos - 1]
        fv = self.create_feature(sen, pos, left_tag)
        fv_id = self.get_feature_id(fv)
        for t in self.tag_dict:
            offset = self.tag_dict[t]*self.feature_length
            score = self.dot(fv_id, offset, flag)
            if(score > max_score):
                max_score = score
                max_tag = t
        #print(max_tag)
        return max_tag
    
    def max_tag_sequence(self, sen, flag):
        tag_sequence = [] # 最终的词性序列
        list_current_tag = [] # 局部最优路径 list of dict
        last_score = {}
        current_score = {}
        current_tag = {}
        max_score = float("-Inf")
        max_tag = ""
        
        # 第一个词 初始化
        left_tag = "START"
        fv = self.create_feature(sen, 0, left_tag)
        fv_id = self.get_feature_id(fv)
        for t in self.tag_dict:
            offset = self.tag_dict[t]*self.feature_length
            temp_score = self.dot(fv_id, offset, flag)
            current_score[t] = temp_score
        
        # 句子长度为1
        sen_len = len(sen.word)
        if sen_len == 1:
            for t in current_score:
                if max_score < current_score[t]:
                    max_score = current_score[t]
                    max_tag = t
            tag_sequence.append(max_tag)
            return tag_sequence
        
        # 后面的词
        for pos in range(1, sen_len):
            last_score = current_score
            current_score = {}
            current_tag = {}
            for t in self.tag_dict: # 当前词性
                max_tag = ""
                max_score = float("-Inf")
                offset = self.tag_dict[t]*self.feature_length
                for left_tag in self.tag_dict: # 前一个词性
                    fv = self.create_feature(sen, pos, left_tag)
                    fv_id = self.get_feature_id(fv)
                    temp_score = self.dot(fv_id, offset, flag) + last_score[left_tag]
                    if max_score < temp_score:
                        max_score = temp_score
                        max_tag = left_tag
                current_score[t] = temp_score
                current_tag[t] = max_tag
            list_current_tag.append(current_tag)
            
        # 最后一个词的max_tag
        max_score = float("-Inf")
        max_tag = ""
        #print(current_score)
        for t in current_score:
            if max_score < current_score[t]:
                max_score = current_score[t]
                max_tag = t
        tag_sequence.append(max_tag)
        
        # 反向回溯
        list_current_tag.reverse()
        for i in range(len(list_current_tag)):
            max_tag = list_current_tag[i][max_tag]
            tag_sequence.insert(0, max_tag)
        
        #print("tag_sequence: " + str(tag_sequence))
        return tag_sequence
    
    def update_w(self, correct_tag_sequence, max_tag_sequence, sen):
        # 第一个词
        left_tag = "START"
        fv = self.create_feature(sen, 0, left_tag)
        fv_id = self.get_feature_id(fv)
        
        correct_tag = correct_tag_sequence[0]
        correct_tag_id = self.tag_dict[correct_tag]
        offset = correct_tag_id*self.feature_length
        for f_id in fv_id:
            self.w[f_id + offset] += 1
            self.v[f_id + offset] += 1
        
        max_tag = max_tag_sequence[0]
        max_tag_id = self.tag_dict[max_tag]
        offset = max_tag_id*self.feature_length
        for f_id in fv_id:
            self.w[f_id + offset] -= 1
            self.v[f_id + offset] -= 1
            
        # 后面的词
        for pos in range(1, len(sen.word)):
            left_tag = sen.tag[pos - 1]
            fv = self.create_feature(sen, pos, left_tag)
            fv_id = self.get_feature_id(fv)
            
            correct_tag = correct_tag_sequence[pos]
            correct_tag_id = self.tag_dict[correct_tag]
            offset = correct_tag_id*self.feature_length
            for f_id in fv_id:
                self.w[f_id + offset] += 1
                self.v[f_id + offset] += 1
            
            max_tag = max_tag_sequence[pos]
            max_tag_id = self.tag_dict[max_tag]
            offset = max_tag_id*self.feature_length
            for f_id in fv_id:
                self.w[f_id + offset] -= 1
                self.v[f_id + offset] -= 1
            
    #def update_v(self):
    #    for t in self.tag_dict:
    #        offset = self.tag_dict[t]*self.feature_length
    #        for f_id in self.feature_values:
    #            self.v[f_id + offset] += self.w[f_id + offset]
            
    def evaluate(self, dataset, flag):
        count = 0
        total_count = 0
        for sen in dataset.sentences:
            # max_tag_sequence = self.max_tag_sequence(sen, flag)
            for pos in range(len(sen.word)):
                # max_tag = max_tag_sequence[pos]
                max_tag = self.max_tag(sen, pos, flag)
                correct_tag = sen.tag[pos]
                if(max_tag == correct_tag):
                    count += 1
                total_count += 1
                #print("total_count:", total_count)
        print(dataset.name +".conll precision:" + str(count / total_count))
        return count, total_count, count / total_count
        
    def online_training(self, max_epochs, flag):
        max_train_precision = 0.0
        max_dev_precision = 0.0
        max_iterator = 0
        print("*******start iteration************")
        for epoch in range(max_epochs):
            print("epoch:" + str(epoch))
            update_times = 0
            for sen in self.train.sentences:
                max_tag_sequence = self.max_tag_sequence(sen, flag)
                correct_tag_sequence = sen.tag
                if(max_tag_sequence != correct_tag_sequence):
                    self.update_w(correct_tag_sequence, max_tag_sequence, sen)
                    update_times += 1
                    
            if(flag != "w"):
                for i in range(self.feature_length*self.tag_length):
                    self.v[i] += update_times*self.w[i]
            
            count_train, total_count_train, pre_train = self.evaluate(self.train, flag)
            count_dev, total_count_dev, pre_dev = self.evaluate(self.dev, flag)
            
            if(pre_train > max_train_precision):
                max_train_precision = pre_train
            if(pre_dev > max_dev_precision):
                max_dev_precision = pre_dev
                max_iterator = epoch
            
        print("**********stop iteration**************")
        if(flag == "w"):
            print("without averaged perceptron:")
        else:
            print("with averaged perceptron")
        print("train.conll max precision:" + str(max_train_precision))
        print("dev.conll max precision:" + str(max_dev_precision) + " in epoch " + str(max_iterator))
            
if __name__ == "__main__":
    starttime = datetime.datetime.now()
    glm_w = global_linear_model()
    glm_w.create_feature_space()
    max_epochs = 15
    glm_w.online_training(max_epochs, "w")
    endtime = datetime.datetime.now()
    print("executing time is "+str((endtime-starttime).seconds)+" s")
    
    """starttime = datetime.datetime.now()
    glm_v = linear_model()
    glm_v.create_feature_space()
    max_epochs = 200
    glm_v.online_training(max_epochs, "v")
    endtime = datetime.datetime.now()
    print("executing time is "+str((endtime-starttime).seconds)+" s")"""