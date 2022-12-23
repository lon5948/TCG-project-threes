/**
 * Framework for Threes! and its variants (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <vector>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include "board.h"
#include "action.h"
#include "weight.h"


static const int featureSize = 6;
static const int featureSize2 = 4;
static const int featureNum = 6;

const std::array<std::array<int, featureSize> ,featureNum> feature = {{
	{{0,1,2,3,4,5}},

	{{4,5,6,7,8,9}},
		
	{{5,6,7,9,10,11}},

	{{9,10,11,13,14,15}},
	
	{{0,1,2,4}},

	{{2,5,6,9}}
}};

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables and a learning rate
 */
class weight_agent : public agent {
public:
	weight_agent(const std::string& args = "") : agent(args), alpha(0.1/48) {
		if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~weight_agent() {
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string& info) {
		std::string res = info; // comma-separated sizes, e.g., "65536,65536"
		for (char& ch : res)
			if (!std::isdigit(ch)) ch = ' ';
		std::stringstream in(res);
		for (size_t size; in >> size; net.emplace_back(size));
	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
	float alpha;
};

/**
 * default random environment, i.e., placer
 * place the hint tile and decide a new hint tile
 */
class random_placer : public random_agent {
public:
	random_placer(const std::string& args = "") : random_agent("name=place role=placer " + args) {
		spaces[0] = { 12, 13, 14, 15 };
		spaces[1] = { 0, 4, 8, 12 };
		spaces[2] = { 0, 1, 2, 3};
		spaces[3] = { 3, 7, 11, 15 };
		spaces[4] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
	}

	virtual action take_action(const board& after) {
		std::vector<int> space = spaces[after.last()];
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;

			int bag[3], num = 0;
			for (board::cell t = 1; t <= 3; t++)
				for (size_t i = 0; i < after.bag(t); i++)
					bag[num++] = t;
			std::shuffle(bag, bag + num, engine);

			board::cell tile = after.hint() ?: bag[--num];
			board::cell hint = bag[--num];

			return action::place(pos, tile, hint);
		}
		return action();
	}

private:
	std::vector<int> spaces[5];
};

/**
 * random player, i.e., slider
 * select a legal action randomly
 */
class random_slider : public random_agent {
public:
	random_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward != -1) return action::slide(op);
		}
		return action();
	}

private:
	std::array<int, 4> opcode;
};

class greedy1step_slider : public random_agent {
public:
	greedy1step_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		int bestOp = -1;
		board::reward bestReward = -1;
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if(reward > bestReward){
				bestReward = reward;
				bestOp = op;
			}
		}
		if (bestReward != -1) return action::slide(bestOp);
		else return action();
	}

private:
	std::array<int, 4> opcode;
};

class greedy2step_slider : public random_agent {
public:
	greedy2step_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		int bestOp = -1;
		board::reward bestReward = -1;
		for (int op : opcode) {
			auto firstBoard = board(before);
			board::reward reward1 = firstBoard.slide(op);
			board::reward bestReward2 = -1;
			for (int op : opcode) {
				board::reward reward2 = board(firstBoard).slide(op);
				if( reward2 > bestReward2 ) bestReward2 = reward2;
			}
			if ( reward1 + bestReward2 > bestReward ) {
				bestReward = reward1 + bestReward2;
				bestOp = op;
			}
		}
		if (bestReward != -1) return action::slide(bestOp);
		else return action();
	}

private:
	std::array<int, 4> opcode;
};

class greedy3step_slider : public random_agent {
public:
	greedy3step_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		int bestOp = -1;
		board::reward bestReward = -1;
		for (int op : opcode) {
			auto firstBoard = board(before);
			board::reward reward1 = firstBoard.slide(op);
			board::reward bestReward2 = -1;
			for (int op : opcode) {
				auto secondBoard = board(firstBoard);
				board::reward reward2 = secondBoard.slide(op);
				board::reward bestReward3 = -1;
				for (int op : opcode) {
					board::reward reward3 = board(secondBoard).slide(op);
					if(reward3 > bestReward3) bestReward3 = reward3;
				}
				if ( reward2 + bestReward3 > bestReward2 ) bestReward2 = reward2 + bestReward3;
			}
			if ( reward1 + bestReward2 > bestReward ) {
				bestReward = reward1 + bestReward2;
				bestOp = op;
			}
		}
		if (bestReward != -1) return action::slide(bestOp);
		else return action();
	}

private:
	std::array<int, 4> opcode;
};

class tdLearning_slider: public weight_agent {
public:
	tdLearning_slider(const std::string& args = "") : weight_agent("name=slide role=slider " + args),
	opcode({ 0, 1, 2, 3 }){}

	virtual void open_episode(const std::string& flag = "") {
        firstFlag = false;
    }

	virtual action take_action(const board& before) {
		auto b = board(before);
		if (!firstFlag) prev = before;
		int bestop = SelectBestOp(b);
		int bestReward = b.slide(bestop);

		if (bestReward != -1) {
			next = b;
			train(bestReward);
			prev = next;
			firstFlag = true;
			return action::slide(bestop);
		}
		else {
			train(bestReward);
			return action();
		}
	}

	unsigned long long int CalculateFeatureIndex(const board& before, int featureIndex) {
		unsigned long long int value = 0;
		auto b = board(before);
		int size;
		if (featureIndex > 3) size = featureSize2;
		else size = featureSize;

		for(int i = 0; i < size; i++){
			value *= 20;
			int row = feature[featureIndex][i] / 4;
			int column = feature[featureIndex][i] % 4;
			value +=  b[row][column];
		}
		return value;
	}

	float CalculateBoardValue(const board& before) {		
		float value = 0.0;
		auto b = board(before);
		for (int r = 0; r < 4; r++) {
			b.rotate_clockwise();
			for (int h = 0; h < 2; h++) {
				b.reflect_horizontal();
				for (int ind = 0; ind < featureNum; ind++) {
					value += net[ind][CalculateFeatureIndex(b, ind)];
				}	
			}
		}	
		return value;
	}

	int SelectBestOp(const board& before) {
		int bestop = -1;
		float maxValue = -1e15;
		for (int op : opcode) {
			auto after = board(before);
			board::reward reward = after.slide(op);
			float boardValue = CalculateBoardValue(after);
			float expectValue = Expectimax(after, op);
			if (reward != -1) {
				if (reward + boardValue + expectValue > maxValue) {
					bestop = op;
					maxValue = reward + boardValue + expectValue;
				}
			}	
		}
		return bestop;
	}

	float Expectimax(const board& after, int op) {
		float result = 0.0;
		int count = 0;
		std::vector<int> pos = {0, 0, 0, 0};
		// up, right, down, left
		if (op == 0) pos = {12, 13, 14, 15};
		else if (op == 1) pos = {0, 4, 8, 12};
		else if (op == 2) pos = {0, 1, 2, 3};
		else if (op == 3) pos = {3, 7, 11, 15};

		for(int ind = 0; ind < 4; ind++){
			if (after(pos[ind]) != 0) continue;
			count++;
			auto b = board(after);
			b.place(pos[ind], b.hint(), b.hint()); 
			float val_max = -1e15;
			for(int i = 0; i < 4; i++){
				auto temp = board(b);
				int reward = temp.slide(i);
				if(reward == -1) continue;
				float v = reward + CalculateBoardValue(temp);
				val_max = std::max(val_max, v);
			}
			if (val_max > -1e15) result += val_max;
		}
		return result/count;
	}

	void train(int reward) {
		double vupdate;
		if (reward == -1) {
			vupdate = alpha * (-CalculateBoardValue(prev));
		}
		else {
			vupdate = alpha * (CalculateBoardValue(next) - CalculateBoardValue(prev) + reward);
		}
		for (int r = 0; r < 4; r++) {
			prev.rotate_clockwise();
			for (int h = 0; h < 2; h++) {
				prev.reflect_horizontal();
				for (int ind = 0; ind < featureNum; ind++) {
					net[ind][CalculateFeatureIndex(prev, ind)] += vupdate;
				}	
			}
		}	
	}

private:
	std::array<int, 4> opcode;
	bool firstFlag = false;
	board prev, next;
};