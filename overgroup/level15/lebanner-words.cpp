// C++17. Standard input is a sequence of little-endian 64-bit integers,
// the strictly increasing base-five encodings of the proposed representatives.
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
using Word = std::uint64_t;
constexpr int length = 15, window = 7;
constexpr int pow5[] = {1,5,25,125,625,3125,15625,78125,390625};
constexpr Word leading = 6103515625ULL;  // 5^14
const int step[] = {0,1,2,1,2};
int transform(int x, int g) {
    return x == 0 ? 0 : 1 + ((x - 1) ^ g);
}
void require(bool b, const char* msg) { if (!b) throw std::runtime_error(msg); }
int canonical_window(int value) {
    int best = value;
    for(int g=0;g<4;++g){
        int x=value, a=0, b=0;
        for(int i=0;i<window;++i){
            int t=transform(x%5,g); x/=5; a=a*5+t; b+=t*pow5[i];
        }
        best=std::min({a,b,best});
    }
    return best;
}
Word closed_count() {
    using State=std::array<int,3>; // state, n0, n1 (n2 = -n0-n1)
    std::map<State,Word> level{{{0,0,0},1}};
    for(int i=0;i<length;++i){
        std::map<State,Word> next;
        for(auto [s,n]:level) for(int x=0;x<5;++x){
            auto t=s; t[0]=(s[0]+step[x])%3;
            if(x>=3){
                if(s[0]<2) ++t[1+s[0]];
                if(t[0]<2) --t[1+t[0]];
            }
            next[t]+=n;
        }
        level.swap(next);
    }
    return level.at({0,0,0});
}
int main(int argc,char** argv) {
    require(argc==2,"supply weights.txt as argument");
    std::ifstream weights(argv[1]); std::string key; int val;
    std::map<int,int> entries;
    while(weights >> key >> val){
        require(key.size()==7 && key.find_first_not_of("01234")==std::string::npos,"bad key");
        int a=std::stoi(key,nullptr,5);
        require(canonical_window(a)==a && entries.emplace(a,val).second,"bad orbit key");
        require(std::abs(val)<=32,"weight out of bounds");
    }
    std::vector<int> table(pow5[7]);
    for(int i=0;i<pow5[7];++i) table[i]=entries.at(canonical_window(i));
    using Maps=std::array<int,8>;
    std::array<std::vector<Maps>,2> halves;
    for(int m=7;m<=8;++m){
        auto& h=halves[m-7]; h.resize(pow5[m]);
        for(int i=0;i<pow5[m];++i) for(int g=0;g<4;++g){
            int t=i,a=0,b=0;
            for(int j=0;j<m;++j){
                int y=transform(t%5,g); t/=5; a=a*5+y; b+=y*pow5[j];
            }
            h[i][g]=b; h[i][4+g]=a;
        }
    }
    Word n=0,volume=0,prev=0,smallest=0; int minscore=481;
    unsigned char bytes[8];
    while(std::cin.read(reinterpret_cast<char*>(bytes),8)){
        Word w=0;
        for(int i=7;i>=0;--i) w=w*256+bytes[i];
        require(w<5*leading && (!n || w>prev),"range or ordering failure");
        prev=w; ++n;
        int digits[21]={}, d=0, count[3]={};
        Word z=w;
        for(int i=14;i>=0;--i){digits[i]=z%5;z/=5;}
        for(int i=0;i<15;++i){
            int x=digits[i], e=(d+step[x])%3;
            if(x>=3){++count[d];--count[e];}
            d=e;
        }
        require(!d && !count[0] && !count[1],"representative is open");
        std::copy(digits,digits+6,digits+15);
        int v=0,score=0;
        for(int j=0;j<6;++j) v=v*5+digits[j];
        for(int j=6;j<21;++j){v=(v%pow5[6])*5+digits[j];score+=table[v];}
        if(score<minscore){minscore=score;smallest=w;}
        const auto &a=halves[1][w/pow5[7]], &b=halves[0][w%pow5[7]];
        int stabilizer=0;
        for(int g=0;g<8;++g){
            z=g<4 ? Word(a[g])*pow5[7]+b[g] : Word(b[g])*pow5[8]+a[g];
            for(int t=0;t<15;++t){
                require(z>=w,"representative is not canonical");
                stabilizer+=(z==w);
                z=(z%leading)*5+z/leading;
            }
        }
        require(stabilizer && 120%stabilizer==0,"bad stabilizer");
        volume+=120/stabilizer;
    }
    require(std::cin.eof() && !std::cin.gcount(),"input is truncated");
    require(volume==closed_count(),"orbit union misses closed words");
    std::cout<<"classes "<<n<<"\nclosed words "<<volume<<"\nminimum integer score "<<minscore
             <<"\nfirst minimizer (base 5, encoded as an integer) "<<smallest<<"\n";
    require(minscore==75,"minimum score check failed");
    return 0;
}
