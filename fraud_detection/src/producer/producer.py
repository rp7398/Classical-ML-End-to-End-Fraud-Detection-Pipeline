import ujson as json
import time,random,uuid
from kafka import KafkaProducer
from datetime import datetime,timedelta
import threading

KAFKA_BOOTSTRAP = '127.0.0.1:29092'
TX_TOPIC = 'transactions'
LABEL_TOPIC = 'lebels'

producer = KafkaProducer(
    bootstrap_servers = KAFKA_BOOTSTRAP,
    value_serializer = lambda v:json.dumps(v).encode('utf-8')
)


def gen_tx():
    user_id = random.randint(1,10000)
    amount = round(random.expovariate(1/50),2)
    device = random.choice(['web','ios','android'])
    ip_hash = random.randint(0,2**32-1)
    is_fraud = random.random() < 0.01

    tx = {
        'tx_id':str(uuid.uuid4()),
        'user_id':user_id,
        'amount':amount,
        'device':device,
        'ip_hash':ip_hash,
        'ts':datetime.utcnow().isoformat(),
        'metadata':{'country':random.choice(['US','IN','BR','NZ','CN'])}
    }

    return tx,is_fraud

def send_labels_later(tx,is_fraud,delay_seconds=3600):
    def send():
        time.sleep(delay_seconds)
        label = {
            'tx_id':tx['tx_id'],
            'user_id':tx['user_id'],
            'amount':tx['amount'],
            'device':tx['device'],
            'ip_hash':tx['ip_hash'],
            'metadata':tx['metadata'],
            'confirmed_fraud':bool(is_fraud),
            'label_ts':datetime.utcnow().isoformat()
        }
        producer.send(LABEL_TOPIC,label)
        producer.flush()

    threading.Thread(target=send,daemon=True).start()

def main(qps=10,delay_seconds=3600):
    while True:
        tx,is_fraud = gen_tx()
        producer.send(TX_TOPIC,tx)
        print(f"Sent tx {tx['tx_id']} (fraud={is_fraud})")

        if random.random() < 0.02 or is_fraud:
            send_labels_later(tx,is_fraud)
        time.sleep(1/qps)

if __name__ == '__main__':
    main(qps=50,delay_seconds=60*30)

