import unittest

from serve.frontend import OutputParser

CALL = '<tool_call><function=read_file><parameter=limit>120</parameter><parameter=path>file.txt</parameter></function></tool_call>'

class ReasoningToolTests(unittest.TestCase):
    def test_calls_in_both_channels_and_split_tags(self):
        for streaming in (False, True):
            for thinking in (False, True):
                for width in (1, 2, 7, 100000):
                    with self.subTest(streaming=streaming, thinking=thinking, width=width):
                        text = ('before' + CALL + 'after</think>answer' + CALL if thinking else 'answer' + CALL)
                        parser = OutputParser(thinking=thinking, stream_tools=streaming)
                        events = []
                        for i in range(0, len(text), width):
                            events += parser.feed(text[i:i+width])
                        events += parser.finish()
                        calls = [e.call for e in events if e.kind == 'tool_call']
                        self.assertEqual(len(calls), 2 if thinking else 1)
                        for call in calls:
                            self.assertEqual(call.name, 'read_file')
                            self.assertEqual(call.arguments, {'limit': 120, 'path': 'file.txt'})
                        self.assertEqual(''.join(e.text for e in events if e.kind == 'reasoning'), 'beforeafter' if thinking else '')
                        self.assertEqual(''.join(e.text for e in events if e.kind == 'content'), 'answer')

    def test_unfinished_reasoning_call_is_not_executable(self):
        p = OutputParser()
        events = p.feed('before<tool_call><function=read_file>') + p.finish()
        self.assertFalse(any(e.kind == 'tool_call' for e in events))
        self.assertEqual(''.join(e.text for e in events if e.kind == 'reasoning'), 'before<tool_call><function=read_file>')

    def test_tag_named_in_prose_inside_reasoning_stays_reasoning(self):
        """The guard from db1559a applied inside reasoning: a bare <tool_call> is only a call when `<function=` follows
        it (after whitespace).  A tag named in prose while thinking is reasoning text, and the call after it is
        still parsed - it is not a malformed call that ends the request."""
        for prose in ('I will use the `<tool_call>` format now.', 'Next I emit a <tool_call> block.',
                      'Two tags <tool_call> and <tool_call>x, then the call.'):
            for streaming in (False, True):
                for width in (1, 2, 7, 100000):
                    with self.subTest(prose=prose, streaming=streaming, width=width):
                        text = f'{prose}\n{CALL}</think>answer'
                        parser = OutputParser(thinking=True, stream_tools=streaming)
                        events = []
                        for i in range(0, len(text), width):
                            events += parser.feed(text[i:i + width])
                        events += parser.finish()
                        calls = [e.call for e in events if e.kind == 'tool_call']
                        self.assertEqual(len(calls), 1)
                        self.assertEqual(calls[0].name, 'read_file')
                        self.assertEqual(''.join(e.text for e in events if e.kind == 'reasoning'),
                                         f'{prose}\n')
                        self.assertEqual(''.join(e.text for e in events if e.kind == 'content'), 'answer')

    def test_reasoning_tag_alone_at_the_end_is_reasoning_text(self):
        for width in (1, 7, 100000):
            with self.subTest(width=width):
                text = 'The format starts with <tool_call>'
                parser = OutputParser(thinking=True)
                events = []
                for i in range(0, len(text), width):
                    events += parser.feed(text[i:i + width])
                events += parser.finish()
                self.assertFalse(any(e.kind == 'tool_call' for e in events))
                self.assertEqual(''.join(e.text for e in events if e.kind == 'reasoning'), text)

if __name__ == '__main__':
    unittest.main()
