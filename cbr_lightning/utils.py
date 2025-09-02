from model_lightning import CBR_RNN, Transformer, LSTM

def load_model(args, vocab_size):
    if args.model == 'CBR_RNN':
        model = CBR_RNN(
        ntoken=vocab_size,
        ninp=args.ninp,
        nhid=args.nhid,
        nheads=args.nheads,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        gumbel_softmax=args.gumbel_softmax,
        criterion='cross_entropy',
        optimizer_type=args.optimizer_type,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler_type,
        seq_len = args.sequence_length,
        compressed_dim = args.compressed_dim,
    )
    elif args.model == 'Transformer':
        model = Transformer(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            lr=args.learning_rate,
            temperature=args.temperature,
            gumbel_softmax=args.gumbel_softmax,
        )
    elif args.model == 'LSTM':
        model = LSTM(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.learning_rate,
        )
    return model