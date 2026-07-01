from datasets import load_dataset


def case_001():
    ds = load_dataset("/home/chenxiang.101/datas/DocVQA", "DocVQA")['validation']
    print(ds)
    
    
if __name__ == '__main__':
    case_001()
