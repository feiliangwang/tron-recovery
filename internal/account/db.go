package account

import (
	"github.com/syndtr/goleveldb/leveldb"
	"github.com/syndtr/goleveldb/leveldb/util"
)

type Db struct {
	db *leveldb.DB
}

func NewAccountDb(path string) (*Db, error) {
	db, err := leveldb.OpenFile(path, nil)
	if err != nil {
		return nil, err
	}
	return &Db{db: db}, nil
}

func (db *Db) Count() int64 {
	var count int64
	iter := db.db.NewIterator(util.BytesPrefix([]byte{0x41}), nil)
	for iter.Next() {
		count++
	}
	return count
}

func (db *Db) IteratorAccount(do func(addr20 []byte)) {
	iter := db.db.NewIterator(util.BytesPrefix([]byte{0x41}), nil)
	for iter.Next() {
		key := iter.Key() // 21 bytes: 0x41 + 20 bytes address
		if len(key) != 21 {
			continue
		}
		addr20 := key[1:] // 去掉 0x41 前缀
		do(addr20)
	}
}

func (db *Db) IsExist(addr20 []byte) bool {
	if _, err := db.db.Get(addr20, nil); err != nil {
		return false
	}
	return true
}
