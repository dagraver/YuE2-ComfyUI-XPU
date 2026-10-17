"""Words cut into the syllables they are sung on: the hand counts the rules were measured on, and how the cuts read."""

from __future__ import annotations

from yue2_comfy import phrasing, syllables

TUNED = """
love 1 heart 1 night 1 day 1 light 1 dream 1 time 1 fire 1 world 1 life 1 eyes 1 rain 1 home 1 road 1 soul 1
never 2 always 2 forever 3 tonight 2 away 2 again 2 alone 2 baby 2 maybe 2 only 2 lonely 2 lovely 2 over 2
beautiful 3 yesterday 3 tomorrow 3 remember 3 together 3 another 3 somebody 3 everybody 4 everything 3 every 2
something 2 sometimes 2 someone 2 nothing 2 anyone 3 anything 3 nobody 3 somewhere 2 nowhere 2 without 2
heaven 2 angel 2 morning 2 evening 2 midnight 2 summer 2 winter 2 river 2 ocean 2 mountain 2 thunder 2 shadow 2
broken 2 falling 2 calling 2 running 2 burning 2 waiting 2 dancing 2 crying 2 dying 2 flying 2 trying 2 being 2
going 2 doing 2 seeing 2 playing 2 saying 2 staying 2 praying 2 believe 2 because 2 before 2 beyond 2 behind 2
between 2 above 2 around 2 across 2 inside 2 outside 2 beside 2 until 2 into 2 upon 2 about 2 below 2
neon 2 video 3 radio 3 piano 3 violin 3 quiet 2 diamond 2 poem 2 poet 2 lion 2 riot 2 idea 3 real 1 really 2
people 2 little 2 middle 2 simple 2 gentle 2 table 2 apple 2 candle 2 candles 2 circles 2 bottles 2 angels 2
whispered 2 listened 2 wanted 2 needed 2 loved 1 lived 1 walked 1 turned 1 burned 1 waited 2 started 2 ended 2
faded 2 called 1 closed 1 opened 2 remembered 3 wondered 2 answered 2 promised 2 changed 1 danced 1 kissed 1
roses 2 faces 2 places 2 voices 2 pieces 2 changes 2 wishes 2 kisses 2 dances 2 houses 2 fires 1 times 1 eyes 1
lives 1 names 1 games 1 flames 1 stones 1 hopes 1 makes 1 takes 1 comes 1 goes 1 does 1 loves 1 moves 1
careful 2 hopeful 2 grateful 2 lifetime 2 sometime 2 homeland 2 movement 2 statement 2 lately 2 safely 2 surely 2
different 2 difference 2 interest 2 memory 3 mystery 3 history 3 family 3 finally 3 suddenly 3 easily 3
heavenly 3 endlessly 3 desperately 3 separate 3 chocolate 2 camera 2 favorite 2 several 2 business 2
emotion 3 devotion 3 nation 2 station 2 passion 2 mission 2 vision 2 illusion 3 confusion 3 forgiveness 3
million 2 billion 2 champion 3 warrior 3 glorious 3 serious 3 curious 3 furious 3 mysterious 4 victorious 4
hour 1 our 1 flower 2 power 2 tower 2 shower 2 higher 2 liar 2 desire 2 tired 1 wire 1 choir 1 prayer 1
create 2 creation 3 reality 4 society 4 variety 4 anxiety 4 science 2 silence 2 violence 3 patience 2
hallelujah 4 chorus 2 rhythm 2 heaven's 2 can't 1 don't 1 won't 1 I'm 1 you're 1 we're 1 they're 1 it's 1
couldn't 2 wouldn't 2 shouldn't 2 didn't 2 isn't 2 wasn't 2 doesn't 2 haven't 2 gonna 2 wanna 2 gotta 2
yeah 1 oh 1 ooh 1 whoa 1 hey 1 through 1 though 1 thought 1 enough 2 laugh 1 breathe 1 breath 1 clothes 1
"""
"""Words songs are made of, with the syllables they are sung on counted by hand: the list the rules were written against."""

HELD = """
shining 2 golden 2 silver 2 window 2 lantern 2 whisper 2 stranger 2 danger 2 stronger 2 longer 2 highway 2
yesterdays 3 nameless 2 homeless 2 careless 2 hopeless 2 timeless 2 useless 2 likely 2 closely 2 bravely 2
loneliness 3 happiness 3 darkness 2 kindness 2 sadness 2 madness 2 freely 2 nicely 2 wisely 2 widely 2
peaceful 2 graceful 2 wasteful 2 shameful 2 basement 2 pavement 2 placement 2 excitement 3 amazement 3
daytime 2 nighttime 2 springtime 2 meantime 2 anything 3 nothing 2 everywhere 3 anywhere 3 elsewhere 2
troubles 2 miracles 3 battles 2 castles 2 whistles 2 smiles 1 miles 1 rules 1 tales 1 holes 1 souls 1
hasn't 2 hadn't 2 aren't 1 weren't 1 mustn't 2 ain't 1 she'll 1 we'll 1 I'll 1 they've 1 you've 1 I've 1
dreamer 2 believer 3 survivor 3 forgotten 3 forgiven 3 awaken 3 horizon 3 paradise 3 memories 3 melody 3
harmony 3 symphony 3 destiny 3 gravity 3 energy 3 enemy 3 lullaby 3 butterfly 3 satellite 3 afternoon 3
whatever 3 whenever 3 wherever 3 however 3 forevermore 4 underneath 3 understand 3 overcome 3 into 2
freedom 2 kingdom 2 wisdom 2 welcome 2 awesome 2 handsome 2 someday 2 somehow 2 goodbye 2 hello 2
wild 1 child 1 mild 1 smile 1 while 1 style 1 tired 1 inspired 2 desired 2 admired 2 retired 2 higher 2
toward 2 towards 2 away 2 okay 2 today 2 always 2 airplane 2 fairytale 3 daydream 2 heartbeat 2 heartache 2
ocean 2 special 2 social 2 precious 2 gracious 2 delicious 3 conscious 2 patient 2 ancient 2 nation 2
champagne 2 machine 2 routine 2 marine 2 magazine 3 gasoline 3 limousine 3 medicine 3 imagine 3 engine 2
blue 1 true 1 through 1 view 1 you 1 new 1 few 1 knew 1 grew 1 flew 1 beautiful 3 beauty 2 duty 2 truly 2
"""
"""More of them, counted by hand after the rules were written and never tuned on."""


def counted(text: str) -> list:
    cells = text.split()
    return [(cells[index], int(cells[index + 1])) for index in range(0, len(cells), 2)]


def cut(text: str) -> list:
    return [syllables.pieces(word) for word in text.split()]


def test_every_word_counted_by_hand_is_counted_the_same():
    """Measured on 2026-10-12: one word wrong in each list, both spelled out since."""
    for text in (TUNED, HELD):
        wrong = [(word, want, syllables.pieces(word)) for word, want in counted(text)
                 if syllables.count(word) != want]
        assert not wrong, wrong


def test_the_pieces_of_a_word_are_the_word_again():
    for word, _want in counted(TUNED) + counted(HELD):
        assert "".join(syllables.pieces(word)) == word
    for word in ("toNIGHT", "caf\u00e9", "na\u00efve", "don\u2019t", "Would\u2019ve", "\u0421\u0415\u0420\u0434\u0446\u0435"):
        assert "".join(syllables.pieces(word)) == word


def test_the_count_load_midi_lays_lines_by_is_left_alone():
    """The words under the notes count for themselves, so no song laid along a MIDI tune moves by a bar."""
    assert [phrasing.syllables(word) for word in ("didn't", "lonely", "something", "candles")] == [1, 3, 3, 1]
    assert [syllables.count(word) for word in ("didn't", "lonely", "something", "candles")] == [2, 2, 2, 2]


def test_a_short_ending_is_sung_with_the_word_or_after_a_consonant_by_itself():
    assert cut("didn't couldn't doesn't haven't") == [["did", "n't"], ["could", "n't"], ["does", "n't"],
                                                      ["have", "n't"]]
    assert cut("can't don't aren't weren't ain't") == [["can't"], ["don't"], ["aren't"], ["weren't"], ["ain't"]]
    assert cut("could've that'll it'd") == [["could", "'ve"], ["that", "'ll"], ["it", "'d"]]
    assert cut("they've we'll she'd you're I'm") == [["they've"], ["we'll"], ["she'd"], ["you're"], ["I'm"]]
    assert cut("there's love's it's heaven's") == [["there's"], ["love's"], ["it's"], ["hea", "ven's"]]
    assert syllables.pieces("didn\u2019t") == ["did", "n\u2019t"]


def test_an_e_before_a_common_ending_is_silent():
    assert cut("lonely something careless lifetime") == [["lone", "ly"], ["some", "thing"], ["care", "less"],
                                                         ["life", "time"]]
    assert cut("sometimes somewhere carefully") == [["some", "times"], ["some", "where"], ["care", "ful", "ly"]]
    assert [syllables.count(word) for word in ("element", "gentleness", "moment", "beside")] == [3, 3, 2, 2]


def test_les_after_a_consonant_and_ue_are_sung():
    assert cut("candle candles little people") == [["can", "dle"], ["can", "dles"], ["lit", "tle"],
                                                   ["peo", "ple"]]
    assert [syllables.count(word) for word in ("smiles", "miles", "rules", "tales")] == [1, 1, 1, 1]
    assert cut("value continue canoe") == [["va", "lue"], ["con", "ti", "nue"], ["ca", "noe"]]
    assert [syllables.count(word) for word in ("tongue", "league", "unique", "argue")] == [1, 1, 2, 2]


def test_words_the_rules_cannot_read_are_spelled_out():
    assert cut("neon quiet idea every") == [["ne", "on"], ["qui", "et"], ["i", "de", "a"], ["eve", "ry"]]
    assert cut("Diamonds heroes NEON") == [["Dia", "monds"], ["he", "roes"], ["NE", "ON"]]
    assert syllables.pieces("naive") == ["na", "ive"]
    assert syllables.pieces("na\u00efve") == ["na", "\u00efve"]


def test_how_the_cuts_between_two_vowels_read():
    assert cut("mother nothing secret children") == [["mo", "ther"], ["no", "thing"], ["se", "cret"],
                                                     ["chil", "dren"]]
    assert cut("singing higher window whisper") == [["sing", "ing"], ["high", "er"], ["win", "dow"],
                                                    ["whis", "per"]]
    assert cut("inspired understand being piano") == [["in", "spired"], ["un", "der", "stand"], ["be", "ing"],
                                                      ["pi", "a", "no"]]


def test_cyrillic_is_a_syllable_a_vowel():
    assert cut("\u043c\u043e\u043b\u043e\u043a\u043e \u0432\u043e\u0439\u043d\u0430 \u043f\u0438\u0441\u044c\u043c\u043e") == [
        ["\u043c\u043e", "\u043b\u043e", "\u043a\u043e"], ["\u0432\u043e\u0439", "\u043d\u0430"],
        ["\u043f\u0438\u0441\u044c", "\u043c\u043e"]]
    assert cut("\u0441\u0435\u043c\u044c\u044f \u043e\u0431\u044a\u0451\u043c \u043c\u043e\u044f") == [
        ["\u0441\u0435", "\u043c\u044c\u044f"], ["\u043e", "\u0431\u044a\u0451\u043c"], ["\u043c\u043e", "\u044f"]]
    assert syllables.count("\u0437\u0434\u0440\u0430\u0432\u0441\u0442\u0432\u0443\u0439") == 2


def test_a_cyrillic_word_with_no_vowel_is_sung_with_its_neighbour():
    assert cut("\u0432 \u043a \u0441 \u0436 \u043b\u044c") == [[], [], [], [], []]
    assert syllables.pieces("\u043c\u043c\u043c") == ["\u043c\u043c\u043c"]
    assert syllables.of_line("\u0432") == 1
    assert syllables.of_line("\u0432 \u0434\u043e\u043c\u0435") == 2


def test_chinese_japanese_and_korean_are_a_syllable_a_character():
    assert syllables.pieces("\u4f60\u597d") == ["\u4f60", "\u597d"]
    assert syllables.count("\u3053\u3093\u306b\u3061\u306f") == 5
    assert syllables.count("\uc0ac\ub791") == 2


def test_a_word_no_rule_reads_is_one_piece():
    assert cut("mm shh") == [["mm"], ["shh"]]
    assert syllables.pieces("\u05e9\u05dc\u05d5\u05dd") == ["\u05e9\u05dc\u05d5\u05dd"]
    assert syllables.pieces("") == []


def test_the_words_of_a_line_are_the_ones_phrasing_counts():
    assert syllables.words("Don't stop -- believin', 24/7!") == ["Don't", "stop", "believin"]
    assert syllables.of_line("Hold the tune and let it ring") == 7
    assert syllables.of_line("... 123 !") == 0
    assert syllables.of_line("") == 0
