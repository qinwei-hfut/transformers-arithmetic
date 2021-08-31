import pdb

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch


class NumPrediction_B_T5(Dataset):
    def __init__(self, path:str, train:bool, num_style:str, question_num:int):
        # path: path of the loaded data
        # train: bool FLAG, it is training data or test data
        # num_style: What convert the num to; 10-base char or str or word
        # question_num: which num is the question?
        assert question_num in [123,13,3]
        assert num_style in [None, 'convert_to_10ebased']

        self.paragraphes = torch.load(path)
        self.train = train
        self.num_style = num_style
        self.question_num = question_num
        self.examples = self.process_paragraphes()

        print('hello')

    def process_paragraphes(self):
        examples = []
        for sentence in self.paragraphes:
            question_pos: int
            num_counter = 0
            answer: str
            for idx, word in enumerate(sentence):

                if Utils.is_number(word):
                    num_counter += 1

                    # num_pos.append(idx)
                    # ori_num.append(word)
                    if self.num_style != None:
                        word = getattr(Utils, self.num_style)(number=word, invert_number=False, split_type=None)
                    sentence[idx] = str(word)

                    if num_counter == 3:
                        answer = sentence[idx]
                        sentence[idx] = '[NUM]'
                        question_pos = idx
                    # converted_num.append(word)


            examples.append({'sentence': sentence, 'question_pos':question_pos, 'answer':answer})
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):

        example = self.examples[item]

        return example['sentence'], example['answer']

        # if self.question_num == 3:
        #     return example['sentence'], example['converted_num'][0:2], example['num_pos'][0:2], example['ori_num'][2], example['num_pos'][2]
        # else:
        #     return example['sentence'], example['converted_num'][0:2], example['num_pos'][0:2], example['ori_num'][2], \
        #            example['num_pos'][2]




    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False



class NumPrediction_B(Dataset):
    def __init__(self, path:str, train:bool, num_style:str, question_num:int):
        # path: path of the loaded data
        # train: bool FLAG, it is training data or test data
        # num_style: What convert the num to; 10-base char or str or word
        # question_num: which num is the question?
        assert question_num in [123,13,3]
        assert num_style in [None, 'convert_to_10ebased']

        self.paragraphes = torch.load(path)
        self.train = train
        self.num_style = num_style
        self.question_num = question_num
        self.examples = self.process_paragraphes()

        print('hello')

    def process_paragraphes(self):
        examples = []
        for sentence in self.paragraphes:
            # example = []
            num_pos = []
            ori_num = []
            converted_num = []
            for idx, word in enumerate(sentence):
                # if self.is_number(word):
                if Utils.is_number(word):
                    sentence[idx] = '[NUM]'
                    num_pos.append(idx)
                    ori_num.append(word)
                    if self.num_style != None:
                        word = getattr(Utils, self.num_style)(number=word, invert_number=False, split_type=None)
                    converted_num.append(word)

            examples.append({'sentence': sentence, 'ori_num': ori_num, 'converted_num': converted_num, 'num_pos': num_pos})
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):

        example = self.examples[item]

        if self.question_num == 3:
            return example['sentence'], example['converted_num'][0:2], example['num_pos'][0:2], example['ori_num'][2], example['num_pos'][2]
        else:
            return example['sentence'], example['converted_num'][0:2], example['num_pos'][0:2], example['ori_num'][2], \
                   example['num_pos'][2]




    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False


class Toy_Static_Num_Dataset(Dataset):
    def __init__(self, digit_top_bound=9, example_num=400000,balance=False):
        print('init: Toy_Static_Num_Dataset')
        print('balance: '+str(balance))

        self.digit_top_bound = digit_top_bound
        self.top_bound = float('1e'+str(digit_top_bound))

        self.example_num = example_num
        if balance == True:
            self.nums_str, self.nums_float = self.generate_balance_num()
        else:
            self.nums_str, self.nums_float = self.generate_num()
        self.nums_digitlist = self.num2digit_list()
        print('finish')

    def __len__(self):
        return len(self.nums_digitlist)

    def __getitem__(self, item):
        return (self.nums_digitlist[item], self.nums_float[item])

    def random_generator(self):
        while(True):
            random_num = torch.rand(1)
            if random_num >= 0.1:
                return random_num
            else:
                continue

    def generate_balance_num(self):
        nums_str = []
        nums_float = []

        # 所以这里生成数据的范围是 0.00x -> 0.99 * 10^b
        for d in range(self.digit_top_bound+1):
            if d == 0:
                for i in range(int(self.example_num / (self.digit_top_bound+1))):
                    num_float = round(torch.rand(1).item(), 2)
                    nums_float.append(num_float)
                    nums_str.append(str(num_float))
            else:
                for i in range(int(self.example_num / (self.digit_top_bound+1))):
                    num_float = round(self.random_generator().item()*float('1e'+str(d)), 2)
                    nums_float.append(num_float)
                    nums_str.append(str(num_float))
        return nums_str,nums_float

    def generate_num(self):
        nums_str = []
        nums_float = []
        for c in range(self.example_num):
            num_float = round(torch.rand(1).item() * self.top_bound,2)
            nums_float.append(num_float)
            nums_str.append(str(num_float))
        return nums_str, nums_float

    def num2digit_list(self):
        numbers_digit_list = []
        for num in self.nums_str:
            num_digit_list = []
            for digit in num:
                # digit_1hot = torch.zeros(11)
                if digit == '.':
                    num_digit_list.append(torch.tensor(10))
                    # digit_1hot[10] = 1
                elif digit in ['0','1','2','3','4','5','6','7','8','9']:
                    num_digit_list.append(torch.tensor(int(digit)))
                    # digit_1hot[int(digit)] = 1
                # num_1hot.append(digit_1hot)
            numbers_digit_list.append(torch.stack(num_digit_list,dim=0))   # shape [seq, embedding]
        return numbers_digit_list

class Static_Num_Pre_Dataset(Dataset):
    def __init__(self, path:str, train:bool, num_style:str, out_style:str):
        # path: path of the loaded data
        # train: bool FLAG, it is training data or test data
        # num_style: What convert the num to; 10-base char or str or word
        # question_num: which num is the question?
        # out_style: 确定数据集的输出的x的格式
        # str_1hot就是普通的计数方式，每一位转成one hot
        assert num_style in [None, 'convert_to_10ebased']
        assert out_style in [None, 'ori_1hot', 'ori_index']

        self.out_style = out_style
        self.paragraphes = torch.load(path)
        self.train = train
        self.num_style = num_style
        self.numbers, self.ori_numbers_str, self.ori_numbers_float = self.collect_numbers()
        self.ori_numbers_1hot = self.num2onehot()
        self.ori_nums_digitlist = self.num2digit_list()

        # print('Static_Num_Pre_Dataset')

    def num2onehot(self):
        numbers_1hot = []
        for num in self.ori_numbers_str:
            num_1hot = []
            for digit in num:
                digit_1hot = torch.zeros(11)
                if digit == '.':
                    digit_1hot[10] = 1
                elif digit in ['0','1','2','3','4','5','6','7','8','9']:
                    digit_1hot[int(digit)] = 1
                num_1hot.append(digit_1hot)
            numbers_1hot.append(torch.stack(num_1hot,dim=0))   # shape [seq, embedding]
        return numbers_1hot


    def num2digit_list(self):
        numbers_digit_list = []
        for num in self.ori_numbers_str:
            num_digit_list = []
            for digit in num:
                # digit_1hot = torch.zeros(11)
                if digit == '.':
                    num_digit_list.append(torch.tensor(10))
                    # digit_1hot[10] = 1
                elif digit in ['0','1','2','3','4','5','6','7','8','9']:
                    num_digit_list.append(torch.tensor(int(digit)))
                    # digit_1hot[int(digit)] = 1
                # num_1hot.append(digit_1hot)
            numbers_digit_list.append(torch.stack(num_digit_list,dim=0))   # shape [seq, embedding]
        return numbers_digit_list


    def collect_numbers(self):
        numbers = []
        ori_numbers_str = []
        ori_numbers_float = []
        for sentence in self.paragraphes:
            # example = []
            for idx, word in enumerate(sentence):
                if Utils.is_number(word):
                    ori_numbers_float.append(float(word))
                    ori_numbers_str.append(word)
                    if self.num_style == 'convert_to_10ebased':
                        word = getattr(Utils, self.num_style)(number=word, invert_number=False, split_type=None)
                    numbers.append(word)

        return numbers, ori_numbers_str, ori_numbers_float

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, item):
        if self.out_style == 'ori_1hot':
            return (self.ori_numbers_1hot[item], self.ori_numbers_float[item])
        if self.out_style == 'ori_index':
            return (self.ori_nums_digitlist[item], self.ori_numbers_float[item])



class Utils:
    def __init__(self):
        pass

    @staticmethod
    def convert_to_10ebased(number: str, split_type: str, invert_number: bool) -> str:
        # signal = None
        # if number[0] == '-':
        #     signal = '-'
        #     number = number[1:]

        output = []
        len_number = len(number)
        point_idx = number.index('.')
        for i, digit in enumerate(number[::-1]):
            if i+1 == len_number-point_idx:
                continue
            elif i+1 < len_number-point_idx:
                i = (i+1) - (len_number-point_idx)
            else:
                i = i - (len_number - point_idx)
            if split_type is None:
                output.append('10e' + str(i))
            elif split_type == 'underscore':
                output.append('10e' + '_'.join(str(i)))
            elif split_type == 'character':
                output.append(' '.join('D' + str(i) + 'E'))
            else:
                raise Exception(f'Wrong split_type: {split_type}')
            output.append(digit)
            # output.append('_')

        # if signal:
        #     output.append(signal)

        # The output is already inverted. If we want it to _not_ be inverted, then we invert it.
        if not invert_number:
            output = output[::-1]
        # output = output[1:]

        return ' '.join(output)

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False


project_path = '/home/qinwei/project/num_prediction/'
data_path = project_path+'train_examples.dat'

'''
num_pre = NumPrediction_B(path=data_path, train=True, num_style='convert_to_10ebased',question_num=3)
print(num_pre.__getitem__(0))
dataloader = DataLoader(dataset=num_pre,batch_size=2,shuffle=True)
for idx, (paragraph, ques_num, ques_num_pos, ans_num, ans_num_pos) in enumerate(dataloader):
    print(idx)
'''

# num_static_dataset = Static_Num_Pre_Dataset(path=data_path,train=True, num_style= 'convert_to_10ebased', out_style='str_1hot')
# dataloader = DataLoader(dataset=num_static_dataset,batch_size=1,shuffle=True)
# for idx, (nums, ori_nums) in enumerate(dataloader):
#     print(idx)