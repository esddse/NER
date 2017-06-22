

from util import *


locs, pers, orgs = [], [], []

def print_percentage(min_loc, max_loc, min_per, max_per, min_org, max_org):
	s_loc, s_per, s_org = 0, 0, 0

	size = len(locs)
	for loc, per, org in zip(locs, pers, orgs):
		if loc >= min_loc and loc <= max_loc:
			s_loc += 1
		if per >= min_loc and per <= max_per:
			s_per += 1
		if org >= min_org and org <= max_org:
			s_org += 1

	print ()
	print (min_loc,'<loc<',max_loc,': ',s_loc/size )
	print (min_per,'<per<',max_per,': ',s_per/size )
	print (min_org,'<org<',max_org,': ',s_org/size )
	print ()

def main():
	sentences, ners, size = read_train_data()

	accloc, accper, accorg = 0, 0, 0
	maxloc, maxper, maxorg = 0, 0, 0
	for ner in ners:
		loc, per, org = 0, 0, 0
		for tag in ner:
			if tag.startswith('B') and tag.endswith('LOC'):
				loc += 1
			elif tag.startswith('B') and tag.endswith('PER'):
				per += 1
			elif tag.startswith('B') and tag.endswith('ORG'):
				org += 1
		locs.append(loc)
		pers.append(per)
		orgs.append(org)

		maxloc = loc if loc > maxloc else maxloc
		maxper = per if per > maxper else maxper
		maxorg = org if org > maxorg else maxorg
		accloc += loc
		accper += per
		accorg += org

	print ('maxloc=',maxloc,' maxper=',maxper,' maxorg=',maxorg)
	print ('avgloc=',accloc/size,' avgper=',accper/size,' avgorg=',accorg/size)
	print_percentage(0, 8, 0, 5, 0, 5)

if __name__ == "__main__":
	main()

